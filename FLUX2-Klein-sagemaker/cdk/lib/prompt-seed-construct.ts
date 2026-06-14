import { readFileSync } from "node:fs";
import { join } from "node:path";

import { Duration, RemovalPolicy } from "aws-cdk-lib";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import { Provider } from "aws-cdk-lib/custom-resources";
import { CustomResource } from "aws-cdk-lib";
import { Construct } from "constructs";

import { PROMPT_OVERRIDE_PK } from "./data-construct";

export interface PromptSeedConstructProps {
  /** The Schedule_Store table to seed the per-effect prompt rows into. */
  readonly scheduleTable: dynamodb.ITable;
}

/**
 * Seeds the initial per-effect prompt into the Schedule_Store on deploy
 * (Requirement: "set the initial prompt for each effect into Dynamo on
 * deployment").
 *
 * Source of truth for the seed is {@link ./effect-prompt-defaults.json} — a
 * snapshot of the catalog defaults in `ui/src/booth/effects.ts`. Each row is
 * written as `pk = PROMPT#OVERRIDE`, `sk = EFFECT#<id>`, `isCustom = false`.
 *
 * Idempotency: the seed uses a conditional `PutItem (attribute_not_exists(pk))`
 * per effect, so re-deploys NEVER overwrite a row an admin has already edited
 * (or a previously seeded row) — only missing rows are created. Admins manage
 * the live value afterwards via the SPA (edit / restore-to-default).
 *
 * Implemented with a single inline Lambda behind a {@link Provider}-backed
 * custom resource (AWS SDK v3, bundled in the Node.js Lambda runtime). The
 * defaults JSON is passed via the resource properties so a change to the
 * snapshot triggers the custom resource to run again on the next deploy.
 */
export class PromptSeedConstruct extends Construct {
  constructor(scope: Construct, id: string, props: PromptSeedConstructProps) {
    super(scope, id);

    // Read the defaults snapshot at synth time so it is embedded in the
    // resource properties (single source of truth = the JSON file).
    const defaultsPath = join(__dirname, "effect-prompt-defaults.json");
    const prompts = JSON.parse(readFileSync(defaultsPath, "utf-8")) as Record<
      string,
      string
    >;

    const onEvent = new lambda.Function(this, "SeedFn", {
      runtime: lambda.Runtime.NODEJS_20_X,
      handler: "index.handler",
      timeout: Duration.minutes(2),
      code: lambda.Code.fromInline(SEED_FN_SOURCE),
    });

    // Least privilege: the seed Lambda may only PutItem on the Schedule_Store,
    // scoped (via the leading-key condition) to the prompt partition.
    onEvent.addToRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ["dynamodb:PutItem"],
        resources: [props.scheduleTable.tableArn],
        conditions: {
          "ForAllValues:StringEquals": {
            "dynamodb:LeadingKeys": [PROMPT_OVERRIDE_PK],
          },
        },
      }),
    );

    const provider = new Provider(this, "SeedProvider", {
      onEventHandler: onEvent,
    });

    const resource = new CustomResource(this, "Seed", {
      serviceToken: provider.serviceToken,
      resourceType: "Custom::EffectPromptSeed",
      properties: {
        TableName: props.scheduleTable.tableName,
        PromptPartitionKey: PROMPT_OVERRIDE_PK,
        // Embedded so a change to the defaults snapshot re-runs the seed.
        Prompts: JSON.stringify(prompts),
      },
    });
    resource.node.addDependency(props.scheduleTable);
    resource.applyRemovalPolicy(RemovalPolicy.DESTROY);
  }
}

/**
 * Inline handler source for the seed Lambda (Node.js 20, AWS SDK v3). On
 * Create/Update it writes each effect's default prompt with a conditional put
 * so existing rows are preserved; on Delete it is a no-op (the table is torn
 * down with the stack).
 */
const SEED_FN_SOURCE = `
const { DynamoDBClient, PutItemCommand } = require("@aws-sdk/client-dynamodb");
const ddb = new DynamoDBClient({});

exports.handler = async (event) => {
  const physicalId = "effect-prompt-seed";
  if (event.RequestType === "Delete") {
    return { PhysicalResourceId: physicalId };
  }

  const props = event.ResourceProperties || {};
  const tableName = props.TableName;
  const pk = props.PromptPartitionKey;
  const prompts = JSON.parse(props.Prompts || "{}");
  const now = new Date().toISOString();

  let created = 0;
  let preserved = 0;
  for (const [effectId, prompt] of Object.entries(prompts)) {
    try {
      await ddb.send(new PutItemCommand({
        TableName: tableName,
        Item: {
          pk: { S: pk },
          sk: { S: "EFFECT#" + effectId },
          effectId: { S: effectId },
          prompt: { S: String(prompt) },
          isCustom: { BOOL: false },
          updatedAt: { S: now },
          updatedBy: { S: "cdk-seed" },
        },
        // Preserve any existing row (admin edit or prior seed).
        ConditionExpression: "attribute_not_exists(pk)",
      }));
      created++;
    } catch (err) {
      if (err && err.name === "ConditionalCheckFailedException") {
        preserved++;
      } else {
        throw err;
      }
    }
  }

  return {
    PhysicalResourceId: physicalId,
    Data: { created: String(created), preserved: String(preserved) },
  };
};
`;
