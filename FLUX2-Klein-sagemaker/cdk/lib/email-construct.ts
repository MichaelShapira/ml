import { Stack } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as ses from "aws-cdk-lib/aws-ses";

/** Props for {@link EmailConstruct}. */
export interface EmailConstructProps {
  /**
   * The email address used as the From of result emails. SES requires this to
   * be a verified identity; this construct creates the identity (verification
   * for an email-address identity is completed by clicking the link SES sends
   * to the address after deploy). Supplied by the root stack from a
   * CfnParameter.
   */
  readonly senderEmail: string;

  /**
   * When `true`, the SES identity is **referenced** rather than created — use
   * this when `senderEmail` is already a verified SES identity in the account
   * (CloudFormation cannot create an identity that already exists). The IAM
   * grant uses the deterministic identity ARN either way, so referencing is
   * fully sufficient.
   *
   * @default false (the construct creates the identity)
   */
  readonly useExistingIdentity?: boolean;
}

/**
 * Email_Service infrastructure: a verified SES sender identity used as the From
 * address when the SPA emails a visitor their photo (SES v2 `SendEmail`).
 *
 * The browser calls SES directly under the Identity-Pool credentials; this
 * construct only provisions (or references) the sender **identity** and exposes
 * its ARN so the auth construct can scope `ses:SendEmail` to it.
 *
 * SES sandbox note: a brand-new account is in the SES sandbox, where BOTH the
 * sender and every recipient must be verified identities. Request SES
 * production access to email arbitrary recipients.
 */
export class EmailConstruct extends Construct {
  /** The verified SES sender identity, when created by this stack. */
  public readonly senderIdentity?: ses.EmailIdentity;

  /** The email address used as the From address. */
  public readonly senderEmail: string;

  constructor(scope: Construct, id: string, props: EmailConstructProps) {
    super(scope, id);

    this.senderEmail = props.senderEmail;

    // Create the identity only when it does not already exist. SES emails a
    // verification link to the address, which the operator must click before
    // sending succeeds. When referencing an existing identity, nothing is
    // created — the deterministic ARN (see senderIdentityArn) is used for IAM.
    if (!props.useExistingIdentity) {
      this.senderIdentity = new ses.EmailIdentity(this, "SenderIdentity", {
        identity: ses.Identity.email(props.senderEmail),
      });
    }
  }

  /**
   * ARN of the sender identity, used to scope `ses:SendEmail` on the Cognito
   * roles. SES identity ARNs are
   * `arn:aws:ses:<region>:<account>:identity/<email>`.
   */
  public get senderIdentityArn(): string {
    return Stack.of(this).formatArn({
      service: "ses",
      resource: "identity",
      resourceName: this.senderEmail,
    });
  }
}

export default EmailConstruct;
