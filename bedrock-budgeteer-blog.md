# Near real-time budget control for multi-tenant Amazon Bedrock environments

by Michael Shapira and Danny Teller | Amazon Web Services

If you run multiple users or tenants on Amazon Bedrock inside a single AWS account, you need budget control that acts in near real time — before a runaway job turns into an unexpected bill. Account-level cost tools cannot identify which user or tenant is overspending right now, and they cannot stop runaway costs before the monthly bill arrives. AWS provides native tools for cloud spending visibility: AWS Cost Explorer and AWS Budgets provide historical trends, threshold-based alerting, and cost allocation across teams. These tools address account-level financial governance. For teams that also need per-user or per-tenant enforcement within a single AWS account, a complementary near real-time layer can further strengthen cost control.

This post introduces Bedrock Budgeteer, an open-source, fully serverless solution that extends AWS-native cost visibility with near real-time budget enforcement, and walks through deployment step by step. Bedrock Budgeteer monitors Amazon Bedrock API calls as they happen and calculates token-based costs within minutes. It automatically enforces per-user or per-tenant spending limits within a single AWS account.

## The challenge: Account-level visibility and per-principal enforcement

AWS Cost Explorer provides up to 13 months of historical data. It supports granular filtering by service, region, tag, and linked account, and includes machine learning-powered forecasting. AWS Budgets extends this with proactive alerting: administrators can define cost or usage thresholds and receive notifications when spending approaches or exceeds those limits. With support for cost budgets, usage budgets, reservation budgets, and Savings Plans budgets, it gives finance and operations teams a structured view of where cloud dollars are going.

For most account-level cost governance scenarios, these tools cover the requirement end to end. The challenge emerges in a specific and increasingly common pattern: multiple users or tenants sharing a single AWS account to invoke Amazon Bedrock foundation models. In this architecture, AWS Cost Explorer and AWS Budgets aggregate Bedrock costs at the account level. The recently introduced IAM principal-based cost allocation attributes costs to the calling IAM identity in the Cost and Usage Report. However, the data still reflects the same up-to-24-hour latency inherent to billing pipelines. Cost attribution exists, but preventing an overage in real time requires a different approach.

Consider a team that has provisioned API keys for 50 internal users, each with a $25 monthly budget. If one user runs an automated batch job overnight that exhausts their budget in three hours, the team will not see that in AWS Cost Explorer until the following day. AWS Budgets will send an alert, but by then the spending has already occurred. The tools report what happened; stopping it from happening requires near real-time visibility at the individual principal level.

Bedrock Budgeteer fills this gap.

## Introducing Bedrock Budgeteer

Bedrock Budgeteer is an open-source solution that automatically monitors and enforces Amazon Bedrock API usage budgets in near real time. It is built entirely on serverless AWS services. There are no servers to provision, no clusters to maintain, and no persistent infrastructure to manage. The solution scales automatically with your workload and requires no operational overhead beyond the initial deployment, which takes 10 to 15 minutes using the AWS Cloud Development Kit (AWS CDK).

The core insight: rather than waiting for billing data to propagate through cost management pipelines, Bedrock Budgeteer intercepts Bedrock API calls at the event level the moment they happen. It calculates cost using the AWS Pricing API and updates budget counters within minutes of each API call, not the following day.

Bedrock Budgeteer supports two primary deployment patterns. In a multi-user scenario, individual developers, data scientists, or internal customers receive their own API key with a dedicated budget. In a multi-tenant scenario, separate tenants (business units, projects, external customers, or application environments) operate within their own spending envelope in the same AWS account. In both cases, Bedrock Budgeteer tracks spending independently per principal and enforces limits automatically.

## How it works: Architecture deep dive

The architecture is event-driven end to end, composed entirely of managed AWS services. Each subsection describes one component in the pipeline.

### Event capture

Amazon Bedrock API calls generate events in AWS CloudTrail. Bedrock Budgeteer configures Amazon EventBridge to capture these events in real time and route them into the processing pipeline. This requires no changes to how users or applications call Amazon Bedrock. The solution requires no proxies, SDK wrappers, or custom endpoints.

### Streaming and cost calculation

Amazon Data Firehose streams the captured events to an AWS Lambda function called the Usage Calculator. This function extracts the token counts from the Bedrock API response metadata. It then retrieves the current per-token pricing for the invoked model from the AWS Pricing API. The result is a precise cost figure for that specific API call, calculated in near real time and always reflecting current model pricing.

### Persistent state

The Usage Calculator writes costs to Amazon DynamoDB, which maintains a running spending total for each user or tenant principal. DynamoDB's single-digit millisecond read/write latency supports keeping budget state current and queryable at any scale.

### Budget monitoring

A dedicated Budget Monitor Lambda function runs on a five-minute schedule using Amazon EventBridge Scheduler. On each execution, it reads the current spending totals from DynamoDB and compares them against the configured budget thresholds for each principal. The system detects budget violations within minutes, not hours.

### Progressive enforcement

When a threshold is crossed, AWS Step Functions orchestrates a graduated response. At the warning threshold (default: 70% of budget), a notification is sent and access continues. At the critical threshold (default: 90%), an escalated alert goes out and a grace period begins. If the principal reaches 100% of their budget, the system updates their IAM policy to suspend Amazon Bedrock access for that specific principal only. Other users and tenants are completely unaffected.

### Notifications

Amazon Simple Notification Service (Amazon SNS) delivers multi-channel alerts at each threshold crossing. Email, SMS, and Slack are supported out of the box. Operations teams receive actionable notifications with context about which principal triggered the alert and how much of their budget has been consumed.

### Automatic restoration

Budget periods refresh automatically on a configurable schedule (default: every 30 days). When a budget period resets, suspended principals are automatically restored to active status with no manual intervention required. Administrators can trigger emergency restoration at any time through a direct DynamoDB update.

## Progressive budget enforcement: A graduated response

One of the most important design decisions in Bedrock Budgeteer is the graduated enforcement model. Suspending access the moment a user crosses a threshold would be disruptive in most environments. Instead, the system applies a three-tier response that gives users and operators time to react before access is interrupted.

The enforcement tiers table summarizes the tiers and their default behavior.

| Tier | Default threshold | Action |
|---|---|---|
| Warning | 70% of budget | Notification sent; access continues normally |
| Critical | 90% of budget | Escalated alert; grace period begins (default: 5 min) |
| Exceeded | 100% of budget | IAM policy updated; Bedrock access suspended for that principal |

At the warning level, the system sends a notification and continues normal operation. This gives users visibility into their consumption trajectory and an opportunity to adjust behavior. At the critical level, the alert escalates and a configurable grace period begins. Only when the budget is fully exhausted and the grace period has elapsed does the system deny Bedrock access for that principal.

Thresholds and timing parameters are configurable through AWS Systems Manager Parameter Store, with no redeployment required. Your team can adjust the warning threshold, extend the grace period, or change the budget refresh cadence through a single `aws ssm put-parameter` command.

## Key capabilities for enterprise deployments

Beyond the core budget enforcement loop, Bedrock Budgeteer includes capabilities that address common operational needs in large-scale deployments.

Most FinOps teams structure budget allocation as a hierarchy: a department holds an overall spending envelope, and individual teams or projects draw from it. Bedrock Budgeteer supports this through pool-based budgets with per-key carve-outs, where teams define a global spending pool and allocate individual budgets to each API key from that pool.

A common governance challenge in shared AWS accounts is API keys created outside the official provisioning process. Bedrock Budgeteer addresses this through rogue key detection: it automatically detects keys that were not provisioned through its CLI tool, tags them, and sends an alert to the operations team. This brings all Bedrock API keys in the account under budget governance, not only the ones formally provisioned.

For financial reporting, AWS Cost Explorer integration provides daily reconciliation between the near real-time DynamoDB spending totals and the authoritative billing data in AWS Cost Explorer. Teams get both the operational near real-time view for enforcement and the financial-grade historical view for reporting and chargeback.

Budget events (threshold crossings, suspensions, restorations, and manual overrides) are logged to Amazon CloudWatch Logs as an audit trail. This supports compliance requirements and provides a complete history of enforcement actions for any principal.

## Quick start commands

### Prerequisites

Before deploying, install these tools:

- Python 3.11 or later
- Node.js 18 or later
- AWS CDK CLI v2 (`npm install -g aws-cdk`)
- AWS CLI configured with credentials that have deployment permissions

### Deployment steps

**Step 1: Clone the repository**

This pulls the source and CDK app onto your machine.

```
git clone https://github.com/teabranch/bedrock-budgeteer.git
cd bedrock-budgeteer
```

Expected: a new `bedrock-budgeteer` directory containing an `app/` folder.

**Step 2: Set up the Python environment**

This creates an isolated environment and installs the CDK app's dependencies.

```
cd app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Expected: your shell prompt is prefixed with `(.venv)` and pip reports a successful install. If `pip install` fails, confirm you are running Python 3.11 or later with `python3 --version`.

**Step 3: Bootstrap CDK in your target account (one-time setup)**

AWS CDK requires a one-time bootstrap that provisions an Amazon S3 bucket and IAM roles for deployment. Find your 12-digit account ID in the AWS Management Console (top-right corner) and choose a Region where Amazon Bedrock is available (for example, `us-east-1` or `us-west-2`):

```
cdk bootstrap aws://123456789012/us-east-1
```

Expected: output ending with `Environment aws://123456789012/us-east-1 bootstrapped.` If you receive a permissions error, confirm that your AWS CLI credentials have `cloudformation:` and `s3:` permissions.

**Step 4: Deploy the solution**

This provisions the full serverless stack. It takes 5–8 minutes in most cases.

```
cdk deploy
```

When prompted to approve IAM changes, enter `y`. Expected: a `deployment complete` message followed by the stack outputs listing resource ARNs and endpoints. If deployment fails, rerun `cdk deploy` — CDK resumes from the last successful resource.

**Verification:** After deployment completes, invoke any Amazon Bedrock model in the account. For example, using the AWS CLI:

```
aws bedrock-runtime invoke-model \
  --model-id anthropic.claude-3-haiku-20240307-v1:0 \
  --body '{"anthropic_version":"bedrock-2023-05-31","max_tokens":50,"messages":[{"role":"user","content":"Hello"}]}' \
  --cli-binary-format raw-in-base64-out \
  /tmp/bedrock-output.json
```

If the model is not yet enabled in your account, enable it under Model access in the Amazon Bedrock console, then retry. Then run this command to check that a spending record appeared:

```
aws dynamodb scan --table-name bedrock-budgeteer-usage --max-items 1
```

The command returns a record with the IAM principal and a cost value within 5 minutes of the invocation.

### Post-deployment: Configure notification endpoints

Amazon SNS delivers budget alerts. Notification channels are configured through AWS Systems Manager Parameter Store. The examples in this section show each supported channel:

```
aws ssm put-parameter \
  --name "/bedrock-budgeteer/notifications/email" \
  --value "your-team@example.com" \
  --type String
```

For SMS:

```
aws ssm put-parameter \
  --name "/bedrock-budgeteer/notifications/sms" \
  --value "+1555123456" \
  --type String
```

For Slack, provide an incoming webhook URL:

```
aws ssm put-parameter \
  --name "/bedrock-budgeteer/notifications/slack-webhook" \
  --value "https://hooks.slack.com/services/T00/B00/xxxx" \
  --type String
```

For detailed setup instructions and troubleshooting, refer to the notification configuration guide in the repository.

After deployment, the system becomes operational. Amazon Bedrock API calls in the account are automatically captured and tracked. New users are provisioned through the `manage_keys.py` CLI. This creates a tagged API key and initializes a budget record in DynamoDB in a single step with no CDK redeployment required.

To provision a new user with a $25 monthly budget, run:

```
python manage_keys.py create-key \
  --user-id alice@example.com \
  --budget 25.00 \
  --period monthly
```

Because the solution is fully serverless, it does not incur baseline infrastructure cost. The solution incurs cost only for AWS service invocations, which scale proportionally with Amazon Bedrock API activity.

## Security and compliance considerations

We designed Bedrock Budgeteer with security as a first-class concern.

IAM roles follow the principle of least privilege, granting each Lambda function and Step Functions state machine only the specific permissions required for its task. The solution stores only the minimum data necessary for budget enforcement: principal identifiers, token counts, and calculated costs. Bedrock Budgeteer does not store sensitive business data, prompt content, or model responses.

AWS encrypts data in DynamoDB and Amazon S3 at rest using AWS-managed keys by default. Organizations with stricter encryption requirements can use customer-managed AWS Key Management Service (AWS KMS) keys. Communications between services use HTTPS/TLS in transit.

For organizations with network isolation requirements, the solution supports optional deployment within an Amazon Virtual Private Cloud (Amazon VPC). The audit trail in CloudWatch Logs provides a complete, tamper-evident record of budget enforcement actions, supporting financial controls and access governance compliance requirements.

## Complementing AWS-native cost management

Bedrock Budgeteer complements AWS-native cost management tools rather than replacing them. AWS Cost Explorer and AWS Budgets remain the authoritative source for account-level financial reporting, forecasting, and chargeback. The recently announced IAM principal-based cost allocation for Amazon Bedrock adds valuable attribution granularity to those reports. These are the right tools for financial governance at the account and organizational level.

Bedrock Budgeteer operates in the operational enforcement layer. It does not ask "how much did we spend last month?" — it asks "is this user about to exceed their budget right now?" Daily reconciliation with AWS Cost Explorer keeps the two views aligned. Teams get both near real-time operational control and the financial-grade reporting their finance teams require.

The comparison table shows how the two approaches work together.

| Capability | AWS Cost Explorer & AWS Budgets | Bedrock Budgeteer |
|---|---|---|
| Account-level cost visibility | Authoritative source | Daily reconciliation |
| Per-user / per-tenant tracking | With IAM tags (up to 24 h delay) | Near real-time (within 5 min) |
| Threshold-based alerting | Account level | Per-principal, multi-tier |
| Automated access enforcement | Not in scope | IAM policy updates |
| Budget period auto-reset | Not in scope | Configurable schedule |
| Historical reporting | 13 months | Via DynamoDB + AWS Cost Explorer |
| Multi-tenant isolation (same account) | Not in scope | Core capability |

## Real-world impact: Tipalti

These results come from a production deployment at Tipalti, a global financial operations platform running multi-user Amazon Bedrock workloads on AWS. All cost figures in this section are based on Tipalti's internal AWS billing data comparing the February 2025 baseline to the March–April 2025 average after deployment.

Before deployment, the team operated five API keys with no per-key spending limits. Monthly Bedrock costs ran between $1,000 and $1,500. After deploying Bedrock Budgeteer and assigning each key a budget of $100 to $200 based on measured usage history, monthly spend stabilized at $850, down from $1,200 in February (a 29% reduction). The enforcement also created a behavioral feedback loop: with defined spending ceilings, developers began optimizing prompt design and model selection to stay within their allocations.

A second deployment covered the team's Amazon Bedrock AgentCore environment, an internal DevOps runtime spanning nine runtimes. Monthly spend there decreased from $500 to $350 (a 30% reduction).

Across both workloads, combined monthly Bedrock spend decreased from approximately $1,700 to $1,200 (roughly 29%), with no changes to application architecture or workload throughput.

## Clean up resources

If you no longer need the Bedrock Budgeteer deployment, you can remove the infrastructure to stop incurring costs. From the project directory with your virtual environment activated, run:

```
cdk destroy
```

When prompted, confirm the deletion. This removes the Lambda functions, DynamoDB tables, Step Functions state machines, EventBridge rules, and associated IAM roles created by the stack. AWS retains CloudWatch Logs log groups by default for audit purposes. To remove them as well, delete them manually via the AWS CLI:

```
aws logs delete-log-group --log-group-name /aws/lambda/bedrock-budgeteer-usage-calculator
aws logs delete-log-group --log-group-name /aws/lambda/bedrock-budgeteer-budget-monitor
```

## Conclusion

As generative AI adoption accelerates, organizations can govern Amazon Bedrock spending at the individual user or tenant level within a single AWS account and in near real time. AWS-native tools cover account-level visibility and alerting. For teams that additionally need per-user or per-tenant granularity updated in near real time, a complementary enforcement layer addresses that specific requirement.

Bedrock Budgeteer provides that layer through a fully serverless, event-driven architecture. It captures Amazon Bedrock API calls as they happen, calculates cost within minutes using live AWS pricing data, and enforces spending limits through a configurable, graduated control system. Suspended principals are restored automatically when their budget period refreshes, keeping the system self-managing and minimizing operational overhead.

Engineering teams, ISVs running multi-tenant AI products, and FinOps teams can all use Bedrock Budgeteer as the near real-time enforcement layer that complements existing AWS cost governance.

We released the solution as open source. The community can explore it, deploy it, and contribute to its continued development.

To get started, visit the Bedrock Budgeteer GitHub repository and follow the deployment guide in the `/docs` folder.

## About the authors

**Michael Shapira** is a solutions architect at Amazon Web Services based in Tel Aviv, Israel, specializing in generative AI and cloud-native architectures.

**Danny Teller** is a solutions architect at Amazon Web Services, focusing on serverless architectures and cost optimization strategies for AI/ML workloads.

## Further reading

For more information, see these resources.

- Track Amazon Bedrock Costs by Caller Identity with IAM-Based Cost Allocation
- AWS Cost Explorer documentation
- AWS Budgets documentation
- Amazon Bedrock documentation
- Bedrock Budgeteer on GitHub
