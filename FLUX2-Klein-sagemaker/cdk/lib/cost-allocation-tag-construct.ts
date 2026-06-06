import {
  custom_resources as cr,
  aws_iam as iam,
} from "aws-cdk-lib";
import { Construct } from "constructs";

export interface CostAllocationTagProps {
  /**
   * The user-defined cost allocation tag KEY to activate (e.g. "AiPhoto").
   * Once active, AWS Cost Explorer can group/filter spend by this tag.
   */
  readonly tagKey: string;
}

/**
 * Activates a user-defined **cost allocation tag** so the stack's spend can be
 * grouped and filtered by it in AWS Cost Explorer (the admin cost panel).
 *
 * AWS does not expose cost-allocation-tag activation through CloudFormation, so
 * this uses an `AwsCustomResource` to call the Billing/Cost Explorer
 * `UpdateCostAllocationTagsStatus` API on deploy.
 *
 * Important caveats (documented, not failures):
 *   - The API is **global** and only succeeds on the **management (payer)
 *     account** of an AWS Organization; member/standalone accounts get an
 *     AccessDenied. The custom resource is therefore configured to **ignore
 *     errors** (`ignoreErrorCodesMatching`) so a deploy on a non-management
 *     account does not fail — the resources are still tagged; only the
 *     Cost-Explorer activation must then be done once, manually, by the payer
 *     account (Billing console → Cost allocation tags → activate "AiPhoto").
 *   - A tag only becomes activatable AFTER tagged resources have reported usage
 *     (it can take up to 24h for the key to appear), so this is best-effort.
 */
export class CostAllocationTagConstruct extends Construct {
  constructor(scope: Construct, id: string, props: CostAllocationTagProps) {
    super(scope, id);

    const call = {
      service: "CostExplorer",
      action: "updateCostAllocationTagsStatus",
      parameters: {
        CostAllocationTagsStatus: [
          { TagKey: props.tagKey, Status: "Active" },
        ],
      },
      physicalResourceId: cr.PhysicalResourceId.of(
        `cost-allocation-tag-${props.tagKey}`,
      ),
      // Tolerate the common non-management-account denial so deploy succeeds.
      ignoreErrorCodesMatching: "AccessDenied.*|.*AccessDenied.*|LinkedAccount.*",
    };

    new cr.AwsCustomResource(this, "ActivateTag", {
      onCreate: call,
      onUpdate: call,
      // No onDelete: leave the activation in place on stack delete.
      policy: cr.AwsCustomResourcePolicy.fromStatements([
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: [
            "ce:UpdateCostAllocationTagsStatus",
            "ce:ListCostAllocationTags",
          ],
          resources: ["*"],
        }),
      ]),
    });
  }
}
