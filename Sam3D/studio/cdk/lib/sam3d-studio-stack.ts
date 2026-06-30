import * as path from "path";
import {
  Stack,
  StackProps,
  CfnOutput,
  Duration,
  RemovalPolicy,
  custom_resources as cr,
} from "aws-cdk-lib";
import { Construct } from "constructs";
import * as cognito from "aws-cdk-lib/aws-cognito";
import * as lambda from "aws-cdk-lib/aws-lambda";
import { NodejsFunction } from "aws-cdk-lib/aws-lambda-nodejs";
import * as apigw from "aws-cdk-lib/aws-apigateway";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as iam from "aws-cdk-lib/aws-iam";
import * as cloudfront from "aws-cdk-lib/aws-cloudfront";
import * as origins from "aws-cdk-lib/aws-cloudfront-origins";

export class Sam3dStudioStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const region = Stack.of(this).region;
    const account = Stack.of(this).account;

    // ---- context / config ----
    const ctx = (k: string, d: string) => (this.node.tryGetContext(k) as string) || d;
    const endpointName = ctx("endpointName", "sam3d-objects-g6e");
    const endpointConfigName = ctx("endpointConfigName", "sam3d-objects-g6e");
    const ioBucketName = ctx("ioBucketName", "") || `sagemaker-${region}-${account}`;
    const inputPrefix = ctx("inputPrefix", "sam3d-inputs/");
    const outputPrefix = ctx("outputPrefix", "sam3d-outputs/");
    const failurePrefix = ctx("failurePrefix", "sam3d-failures/");
    const adminEmail = ctx("adminEmail", "admin@example.com");
    const visitorEmail = ctx("visitorEmail", "visitor@example.com");

    const endpointArn = `arn:aws:sagemaker:${region}:${account}:endpoint/${endpointName.toLowerCase()}`;
    const endpointConfigArn = `arn:aws:sagemaker:${region}:${account}:endpoint-config/${endpointConfigName.toLowerCase()}`;
    const ioBucketArn = `arn:aws:s3:::${ioBucketName}`;

    // ===================================================================
    // 1) Cognito — User Pool with a custom:role attribute, NO self sign-up
    // ===================================================================
    const userPool = new cognito.UserPool(this, "UserPool", {
      userPoolName: "sam3d-studio",
      selfSignUpEnabled: false, // only provisioned users — no public sign-up
      signInAliases: { username: true, email: true },
      autoVerify: { email: false },
      standardAttributes: { email: { required: true, mutable: true } },
      customAttributes: {
        // rides in the JWT as "custom:role" (admin | visitor)
        role: new cognito.StringAttribute({ minLen: 1, maxLen: 16, mutable: true }),
      },
      passwordPolicy: {
        minLength: 12,
        requireLowercase: true,
        requireUppercase: true,
        requireDigits: true,
        requireSymbols: true,
      },
      accountRecovery: cognito.AccountRecovery.EMAIL_ONLY,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    const userPoolClient = userPool.addClient("WebClient", {
      userPoolClientName: "sam3d-studio-web",
      generateSecret: false, // public SPA client
      authFlows: { userSrp: true, userPassword: true },
      accessTokenValidity: Duration.hours(1),
      idTokenValidity: Duration.hours(1),
      refreshTokenValidity: Duration.days(30),
      preventUserExistenceErrors: true,
    });

    // Two provisioned users. Passwords are set OUT-OF-BAND by deploy.sh
    // (admin-set-user-password) so they never live in CloudFormation.
    const adminUser = new cognito.CfnUserPoolUser(this, "AdminUser", {
      userPoolId: userPool.userPoolId,
      username: "admin",
      messageAction: "SUPPRESS",
      userAttributes: [
        { name: "custom:role", value: "admin" },
        { name: "email", value: adminEmail },
        { name: "email_verified", value: "true" },
      ],
    });
    const visitorUser = new cognito.CfnUserPoolUser(this, "VisitorUser", {
      userPoolId: userPool.userPoolId,
      username: "visitor",
      messageAction: "SUPPRESS",
      userAttributes: [
        { name: "custom:role", value: "visitor" },
        { name: "email", value: visitorEmail },
        { name: "email_verified", value: "true" },
      ],
    });
    adminUser.node.addDependency(userPool);
    visitorUser.node.addDependency(userPool);

    // ===================================================================
    // 2) Static SPA hosting — private S3 + CloudFront (OAC)
    // ===================================================================
    const siteBucket = new s3.Bucket(this, "SpaBucket", {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    const distribution = new cloudfront.Distribution(this, "Distribution", {
      defaultBehavior: {
        origin: origins.S3BucketOrigin.withOriginAccessControl(siteBucket),
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        cachePolicy: cloudfront.CachePolicy.CACHING_OPTIMIZED,
      },
      defaultRootObject: "index.html",
      errorResponses: [
        { httpStatus: 403, responseHttpStatus: 200, responsePagePath: "/index.html" },
        { httpStatus: 404, responseHttpStatus: 200, responsePagePath: "/index.html" },
      ],
      comment: "SAM3D Studio SPA",
    });
    const corsOrigin = `https://${distribution.distributionDomainName}`;
    // Origins allowed to call the API / presign-upload to S3. CloudFront is the
    // real one; localhost:5173 lets `npm run dev` talk to the live backend.
    const allowedOrigins = [corsOrigin, "http://localhost:5173"];

    // ===================================================================
    // 3) Lambdas (each with a least-privilege role)
    // ===================================================================
    const lambdaDir = path.join(__dirname, "..", "lambda");
    const commonEnv = {
      IO_BUCKET: ioBucketName,
      INPUT_PREFIX: inputPrefix,
      OUTPUT_PREFIX: outputPrefix,
      FAILURE_PREFIX: failurePrefix,
      ENDPOINT_NAME: endpointName,
      ENDPOINT_CONFIG_NAME: endpointConfigName,
      PRESIGN_TTL: "300",
    };

    // ---- userApi: upload-url, generate, result, endpoint status ----
    const userApiFn = new NodejsFunction(this, "UserApiFn", {
      entry: path.join(lambdaDir, "user-api.ts"),
      runtime: lambda.Runtime.NODEJS_20_X,
      memorySize: 256,
      timeout: Duration.seconds(29),
      environment: commonEnv,
      bundling: { minify: true, externalModules: [] },
    });
    // S3: write inputs, read outputs/failures (prefix-scoped)
    userApiFn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["s3:PutObject"],
        resources: [`${ioBucketArn}/${inputPrefix}*`],
      })
    );
    userApiFn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["s3:GetObject"],
        resources: [`${ioBucketArn}/${outputPrefix}*`, `${ioBucketArn}/${failurePrefix}*`],
      })
    );
    userApiFn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["s3:ListBucket"],
        resources: [ioBucketArn],
        conditions: { StringLike: { "s3:prefix": [`${outputPrefix}*`, `${failurePrefix}*`] } },
      })
    );
    // SageMaker: invoke + read status of the one endpoint
    userApiFn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["sagemaker:InvokeEndpointAsync", "sagemaker:DescribeEndpoint"],
        resources: [endpointArn],
      })
    );

    // ---- adminApi: start/stop (admin claim enforced in code) ----
    const adminApiFn = new NodejsFunction(this, "AdminApiFn", {
      entry: path.join(lambdaDir, "admin-api.ts"),
      runtime: lambda.Runtime.NODEJS_20_X,
      memorySize: 256,
      timeout: Duration.seconds(29),
      environment: commonEnv,
      bundling: { minify: true, externalModules: [] },
    });
    adminApiFn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: [
          "sagemaker:CreateEndpoint",
          "sagemaker:DeleteEndpoint",
          "sagemaker:DescribeEndpoint",
          "sagemaker:AddTags",
        ],
        // CreateEndpoint authorizes against BOTH the endpoint and the
        // endpoint-config resources, so the config ARN must be included or the
        // call is denied (the other actions simply ignore the extra ARN).
        resources: [endpointArn, endpointConfigArn],
      })
    );
    adminApiFn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["sagemaker:DescribeEndpointConfig"],
        resources: [endpointConfigArn],
      })
    );

    // ===================================================================
    // 4) REST API + Cognito authorizer (every route requires a JWT)
    // ===================================================================
    const api = new apigw.RestApi(this, "Api", {
      restApiName: "sam3d-studio",
      deployOptions: { stageName: "prod", throttlingRateLimit: 20, throttlingBurstLimit: 40 },
      defaultCorsPreflightOptions: {
        allowOrigins: allowedOrigins,
        allowMethods: ["GET", "POST", "OPTIONS"],
        allowHeaders: ["Authorization", "Content-Type"],
      },
    });

    const authorizer = new apigw.CognitoUserPoolsAuthorizer(this, "JwtAuthorizer", {
      cognitoUserPools: [userPool],
    });
    const authOpts: apigw.MethodOptions = {
      authorizer,
      authorizationType: apigw.AuthorizationType.COGNITO,
    };

    const userInt = new apigw.LambdaIntegration(userApiFn);
    const adminInt = new apigw.LambdaIntegration(adminApiFn);

    api.root.addResource("upload-url").addMethod("POST", userInt, authOpts);
    api.root.addResource("generate").addMethod("POST", userInt, authOpts);
    api.root.addResource("result").addMethod("GET", userInt, authOpts);
    const endpointRes = api.root.addResource("endpoint");
    endpointRes.addMethod("GET", userInt, authOpts); // status — any authed user
    endpointRes.addResource("start").addMethod("POST", adminInt, authOpts); // admin
    endpointRes.addResource("stop").addMethod("POST", adminInt, authOpts); // admin

    // ===================================================================
    // 5) CORS on the EXISTING I/O bucket (presigned PUT/GET from the SPA)
    // ===================================================================
    new cr.AwsCustomResource(this, "IoBucketCors", {
      onUpdate: {
        service: "S3",
        action: "putBucketCors",
        parameters: {
          Bucket: ioBucketName,
          CORSConfiguration: {
            CORSRules: [
              {
                AllowedOrigins: allowedOrigins,
                AllowedMethods: ["PUT", "GET", "HEAD"],
                AllowedHeaders: ["*"],
                ExposeHeaders: ["ETag"],
                MaxAgeSeconds: 3000,
              },
            ],
          },
        },
        physicalResourceId: cr.PhysicalResourceId.of(`cors-${ioBucketName}`),
      },
      policy: cr.AwsCustomResourcePolicy.fromStatements([
        new iam.PolicyStatement({ actions: ["s3:PutBucketCors"], resources: [ioBucketArn] }),
      ]),
    });

    // ===================================================================
    // 6) Outputs (consumed by deploy.sh to build the SPA config)
    // ===================================================================
    new CfnOutput(this, "Region", { value: region });
    new CfnOutput(this, "UserPoolId", { value: userPool.userPoolId });
    new CfnOutput(this, "UserPoolClientId", { value: userPoolClient.userPoolClientId });
    new CfnOutput(this, "ApiUrl", { value: api.url });
    new CfnOutput(this, "DistributionDomain", { value: corsOrigin });
    new CfnOutput(this, "DistributionId", { value: distribution.distributionId });
    new CfnOutput(this, "SpaBucketName", { value: siteBucket.bucketName });
    new CfnOutput(this, "IoBucket", { value: ioBucketName });
  }
}
