# Requirements Document

## Introduction

The AI Photo Booth is an interactive kiosk application for an office lobby. A visitor walks up to a portrait-oriented touch screen, captures a photo of themselves with an attached webcam, and applies AI effects to that photo. Effects either replace the background (for example, a spaceship interior) or transform the subject's appearance (for example, rendered as a viking warrior). The selected effect drives a prompt and the captured photo as a reference image to an existing FLUX.2 [klein] 9B SageMaker asynchronous inference endpoint; the transformed image is returned to the visitor.

The application has no per-request backend API. The React single-page application calls AWS services directly from the browser using temporary AWS credentials vended by an Amazon Cognito identity pool, with authorization enforced by AWS IAM roles. Standard signed-in users assume an Authenticated_Role; administrators assume an Admin_Role mapped from a Cognito group.

The entire application is gated behind Amazon Cognito authentication. Users whose Cognito `profile` custom attribute equals `ADMIN` additionally gain access to an Admin UI for managing the SageMaker endpoint: listing and selecting endpoints, starting and stopping (deleting) an endpoint, viewing and refreshing endpoint status, and scheduling endpoint working hours through a calendar interface backed by DynamoDB. Admin-only AWS operations are authorized at the IAM layer rather than by a backend.

A Scheduler_Function (an AWS Lambda) runs on a recurring cadence to automatically start and stop the endpoint according to the defined working hours, providing cost control.

All infrastructure is provisioned with AWS CDK. The CDK application accepts the initial admin user and an initial standard user as template parameters, assigns a DELETE removal policy to all resources, and deploys the React UI to a private S3 bucket served through CloudFront.

This document defines the functional and non-functional requirements for the photo booth capture and effect flow, the predefined effect option sets, direct-to-AWS asynchronous generation handling, authentication and IAM-based admin gating, admin endpoint management, the DynamoDB-backed scheduling calendar, the automated endpoint scheduler, the CDK infrastructure, and the portrait-mode touch UI constraints.

## Glossary

- **Photo_Booth_App**: The complete client-facing React single-page application running on the kiosk touch screen, including the capture flow, effect selection, and the client-side modules that call AWS services directly using the AWS SDK for JavaScript v3.
- **Capture_Module**: The component of the Photo_Booth_App responsible for displaying the live webcam feed and capturing a still photo.
- **Effect_Selector**: The component of the Photo_Booth_App that presents background and person effect options and records the visitor's selection.
- **Effect_Option**: A predefined transformation choice belonging to either the background category or the person category. Each Effect_Option maps to a prompt template.
- **Generation_Service**: A client-side module of the Photo_Booth_App that submits requests to the FLUX2_Endpoint by calling AWS services directly with the AWS SDK for JavaScript v3 and returns the resulting image to the capture flow.
- **FLUX2_Endpoint**: The existing SageMaker asynchronous inference endpoint named `flux2-klein-9b-g6e2` serving the FLUX.2-klein-9B model.
- **Async_Request**: A single JSON inference request submitted to the FLUX2_Endpoint via the S3 input location and asynchronous invocation.
- **Output_Location**: The S3 location where the FLUX2_Endpoint writes the generated `image/png` result for a given Async_Request.
- **Failure_Location**: The S3 location where the FLUX2_Endpoint writes a failure record for a given Async_Request.
- **Auth_Service**: The Amazon Cognito user pool and associated app client that authenticate users of the Photo_Booth_App.
- **Identity_Pool**: The Amazon Cognito identity pool that exchanges a Cognito user pool token for temporary AWS credentials used by the Photo_Booth_App to call AWS services directly.
- **Authenticated_Role**: The AWS IAM role assumed by a Standard_User through the Identity_Pool, granting the least-privilege permissions required for the capture and effect flow.
- **Admin_Role**: The AWS IAM role assumed by an Admin_User through the Identity_Pool, granting the permissions required for endpoint management and schedule data changes in addition to the capture and effect flow.
- **Profile_Attribute**: The Cognito custom user attribute named `profile`. A value of `ADMIN` designates an administrator.
- **Admin_User**: A signed-in user whose Profile_Attribute value equals `ADMIN`.
- **Standard_User**: A signed-in user whose Profile_Attribute value is absent or not equal to `ADMIN`.
- **Admin_UI**: The administrator-only interface, exposed as a tab, for endpoint management and scheduling.
- **Endpoint_Manager**: A client-side module of the Photo_Booth_App that lists, selects, starts, deletes, and reports the status of SageMaker endpoints, and that reads and writes schedule data, by calling SageMaker management APIs and DynamoDB directly under the Admin_Role.
- **Schedule_Calendar**: The Admin_UI calendar component for defining endpoint working hours per day.
- **Working_Hours**: A start time and end time for a specific day during which an endpoint is intended to be running.
- **Schedule_Store**: The DynamoDB table that persists Working_Hours data.
- **Scheduler_Function**: The AWS Lambda function that runs on a recurring cadence to reconcile the running state of the FLUX2_Endpoint against the defined Working_Hours.
- **Infrastructure_Stack**: The AWS CDK application and stacks that provision all cloud resources for the system.
- **Web_Distribution**: The Amazon CloudFront distribution that serves the Photo_Booth_App from a private S3 bucket.
- **UI_Bucket**: The private Amazon S3 bucket that stores the built React UI assets.

## Requirements

### Requirement 1: Start Screen

**User Story:** As a visitor, I want a simple start screen, so that I can begin using the photo booth with a single tap.

#### Acceptance Criteria

1. WHILE no capture session is active, THE Photo_Booth_App SHALL display a start screen containing a single primary Start control.
2. WHEN the visitor taps the Start control, THE Photo_Booth_App SHALL transition to the live camera view.
3. THE Photo_Booth_App SHALL render the start screen in portrait orientation with the Start control sized for touch interaction.

### Requirement 2: Live Camera View

**User Story:** As a visitor, I want to see a live feed of myself, so that I can frame my shot before taking the photo.

#### Acceptance Criteria

1. WHEN the live camera view is displayed, THE Capture_Module SHALL render the video stream from the attached webcam.
2. WHILE the live camera view is displayed and capture is available, THE Capture_Module SHALL display a Take Photo control.
3. WHILE photo capture is temporarily unavailable, THE Capture_Module SHALL hide the Take Photo control and SHALL display a message indicating capture is temporarily unavailable.
4. IF the attached webcam is unavailable or access is denied, THEN THE Capture_Module SHALL display an error message identifying that the camera cannot be accessed and SHALL provide a control to retry camera access.
5. THE Capture_Module SHALL render the live camera view in portrait orientation.

### Requirement 3: Photo Capture

**User Story:** As a visitor, I want to capture a still photo, so that I can apply effects to it.

#### Acceptance Criteria

1. WHEN the visitor taps the Take Photo control, THE Capture_Module SHALL capture a still image from the current webcam frame.
2. WHEN a still image is captured, THE Photo_Booth_App SHALL display the captured photo to the visitor.
3. WHILE the captured photo is displayed, THE Photo_Booth_App SHALL display a Reset control and a Continue control.
4. THE Capture_Module SHALL encode the captured image as a base64-encoded PNG or JPEG string for later submission to the Generation_Service.

### Requirement 4: Reset and Continue After Capture

**User Story:** As a visitor, I want to retake or accept my photo, so that I am satisfied before applying effects.

#### Acceptance Criteria

1. WHEN the visitor taps the Reset control, THE Photo_Booth_App SHALL discard the captured photo and return to the live camera view.
2. WHEN the visitor taps the Continue control, THE Photo_Booth_App SHALL transition to the effect selection view and SHALL retain the captured photo.
3. WHILE the effect selection view is displayed, THE Photo_Booth_App SHALL display the captured photo alongside the effect options.

### Requirement 5: Background Effect Options

**User Story:** As a visitor, I want to choose a new background, so that my photo appears in a different setting.

#### Acceptance Criteria

1. WHEN the effect selection view is displayed, THE Effect_Selector SHALL present exactly 6 background Effect_Options.
2. THE Effect_Selector SHALL include background Effect_Options for "Spaceship interior", "Roman colosseum", and "Tropical background".
3. THE Effect_Selector SHALL include three additional background Effect_Options: "Snowy mountain peak", "Neon city street at night", and "Enchanted forest".
4. THE Effect_Selector SHALL render each background Effect_Option as a touch-sized control with a descriptive label.
5. WHEN the visitor selects a background Effect_Option, THE Effect_Selector SHALL record the selected option as the active transformation request.
6. THE Effect_Selector SHALL display the 6 background Effect_Options and the 6 person Effect_Options simultaneously in the effect selection view, for a total of 12 visible options.

### Requirement 6: Person Effect Options

**User Story:** As a visitor, I want to transform how I look, so that my photo depicts me as a different character.

#### Acceptance Criteria

1. WHEN the effect selection view is displayed, THE Effect_Selector SHALL present exactly 6 person Effect_Options.
2. THE Effect_Selector SHALL include person Effect_Options for "Viking warrior" and "Roman emperor".
3. THE Effect_Selector SHALL include four additional person Effect_Options: "Astronaut", "Renaissance noble", "Cyberpunk hacker", and "Medieval knight".
4. THE Effect_Selector SHALL render each person Effect_Option as a touch-sized control with a descriptive label.
5. WHEN the visitor selects a person Effect_Option, THE Effect_Selector SHALL record the selected option as the active transformation request.
6. WHILE a transformation request is being processed, THE Effect_Selector SHALL ignore additional Effect_Option selections and SHALL retain the first selection until processing completes.

### Requirement 7: Effect Prompt Mapping

**User Story:** As a visitor, I want my chosen effect to produce a relevant result, so that the transformation matches the option I selected.

#### Acceptance Criteria

1. THE Effect_Selector SHALL map each Effect_Option to a predefined prompt string.
2. WHEN the visitor selects an Effect_Option, THE Generation_Service SHALL construct an Async_Request whose `inputs` field is set to the prompt string mapped to the selected Effect_Option.
3. WHEN the visitor selects an Effect_Option, THE Generation_Service SHALL include the captured photo as a base64-encoded reference image in the `images` field of the Async_Request.
4. THE Generation_Service SHALL set `num_inference_steps` to a value within the inclusive range 4 to 20 and `guidance_scale` to a value within the inclusive range 1 to 10 in the Async_Request, so that low values that degrade output quality are avoided.

### Requirement 8: Asynchronous Generation Submission

**User Story:** As a visitor, I want my transformation request to be processed reliably, so that I receive a result.

#### Acceptance Criteria

1. WHEN the Generation_Service submits an Async_Request, THE Generation_Service SHALL write the request JSON to the S3 input prefix and call SageMaker Runtime InvokeEndpointAsync directly using the temporary AWS credentials.
2. WHILE an Async_Request is pending, THE Generation_Service SHALL poll the Output_Location and the Failure_Location directly using S3 HEAD and GetObject operations.
3. WHEN the FLUX2_Endpoint writes a result image to the Output_Location, THE Generation_Service SHALL read the `image/png` result directly from S3 and return it to the capture flow.
4. IF the FLUX2_Endpoint writes a failure record to the Failure_Location, THEN THE Generation_Service SHALL return an error response identifying that generation failed.
5. IF no result and no failure are available within 120 seconds of submission, THEN THE Generation_Service SHALL return a timeout error response, enforced client-side, to the capture flow.

### Requirement 9: Loading and Progress State

**User Story:** As a visitor, I want to see that my photo is being processed, so that I know the system is working during the multi-second wait.

#### Acceptance Criteria

1. WHEN a transformation request is submitted, THE Photo_Booth_App SHALL display a loading state indicating that the image is being generated.
2. WHILE the transformation request is pending, THE Photo_Booth_App SHALL keep the loading state visible.
3. WHEN a successful transformed image is received, THE Photo_Booth_App SHALL replace the loading state with the transformed image such that the loading state and the transformed image are never displayed at the same time.
4. IF the Generation_Service returns an error or timeout response, THEN THE Photo_Booth_App SHALL replace the loading state with an error message, SHALL withhold any transformed image, and SHALL provide a control to retry or return to the effect selection view.

### Requirement 10: Result Display and Session Restart

**User Story:** As a visitor, I want to see my transformed photo and start over, so that the booth is ready for me or the next person.

#### Acceptance Criteria

1. WHEN the transformed image is received, THE Photo_Booth_App SHALL display the transformed image in portrait orientation.
2. WHILE the transformed image is displayed, THE Photo_Booth_App SHALL display a control to start a new session.
3. WHILE the transformed image is displayed, THE Photo_Booth_App SHALL display a notice that the image is AI-generated.
4. WHEN the visitor taps the start-a-new-session control, THE Photo_Booth_App SHALL discard the captured photo and transformed image and SHALL return to the start screen.
5. WHEN the Photo_Booth_App returns to the start screen by any path, THE Photo_Booth_App SHALL discard any captured photo and transformed image from the prior session.

### Requirement 11: Authentication

**User Story:** As the office operator, I want the application gated behind sign-in, so that only authorized users can access the booth and admin functions.

#### Acceptance Criteria

1. WHILE no user is authenticated, THE Photo_Booth_App SHALL restrict access to the capture flow and SHALL present a sign-in interface provided by the Auth_Service.
2. WHEN a user submits valid credentials, THE Auth_Service SHALL authenticate the user, THE Identity_Pool SHALL vend temporary AWS credentials, and THE Photo_Booth_App SHALL grant access to the capture flow.
3. IF a user submits invalid credentials, THEN THE Auth_Service SHALL deny access and THE Photo_Booth_App SHALL display an authentication error message.
4. WHEN an authenticated user signs out, THE Photo_Booth_App SHALL discard the active session and the temporary AWS credentials and SHALL return to the sign-in interface.

### Requirement 12: Silent Authentication and Credential Refresh

**User Story:** As a visitor, I want my session to continue without repeated sign-ins, so that an expired token or credential does not interrupt me while the booth is in use.

#### Acceptance Criteria

1. IF a Cognito access token or ID token has expired AND the refresh token remains valid, THEN THE Photo_Booth_App SHALL silently refresh the Cognito token before surfacing an authentication error to the visitor.
2. IF the temporary AWS credentials vended by the Identity_Pool have expired AND the refresh token remains valid, THEN THE Photo_Booth_App SHALL silently obtain replacement temporary AWS credentials before surfacing an authentication error to the visitor.
3. WHILE the refresh token remains valid, THE Photo_Booth_App SHALL grant continued access to the capture flow without prompting the visitor to sign in again.
4. IF the refresh token is no longer valid, THEN THE Photo_Booth_App SHALL present the sign-in interface to re-authenticate the visitor.

### Requirement 13: Connection Status Indicator

**User Story:** As a visitor, I want a simple connection indicator that does not reveal identity, so that no personal identity is shown on the shared kiosk.

#### Acceptance Criteria

1. THE Photo_Booth_App SHALL display a connection indicator with exactly two states: "connected" and "disconnected".
2. WHILE the user is authenticated with valid temporary AWS credentials, THE Photo_Booth_App SHALL display the connection indicator in the "connected" state.
3. WHILE the user is not authenticated OR valid temporary AWS credentials are unavailable, THE Photo_Booth_App SHALL display the connection indicator in the "disconnected" state.
4. THE Photo_Booth_App SHALL exclude the signed-in user's username and identity from every screen of the user interface.

### Requirement 14: Admin Gating

**User Story:** As an administrator, I want the Admin UI visible only to admins and admin operations enforced by IAM, so that standard visitors cannot access endpoint management.

#### Acceptance Criteria

1. WHILE no user is authenticated OR the signed-in user is a Standard_User, THE Photo_Booth_App SHALL hide the Admin_UI tab.
2. WHILE the signed-in user is an Admin_User, THE Photo_Booth_App SHALL display the Admin_UI tab.
3. THE Photo_Booth_App SHALL determine administrator status by reading the Profile_Attribute from the authenticated user's Cognito profile and comparing the value to `ADMIN`.
4. WHILE the signed-in user belongs to the Cognito admin group, THE Identity_Pool SHALL vend temporary AWS credentials for the Admin_Role.
5. WHILE the signed-in user is a Standard_User, THE Identity_Pool SHALL vend temporary AWS credentials for the Authenticated_Role, which excludes SageMaker management permissions and Schedule_Store write permissions.
6. IF a Standard_User attempts an admin AWS operation, THEN the targeted AWS service SHALL deny the operation under the Authenticated_Role policy.

### Requirement 15: List and Select Endpoints

**User Story:** As an administrator, I want to see and select SageMaker endpoints, so that I can manage the one serving the booth.

#### Acceptance Criteria

1. WHEN the Admin_User opens the Admin_UI, THE Endpoint_Manager SHALL call SageMaker ListEndpoints directly under the Admin_Role and SHALL return the list of SageMaker endpoints in the configured AWS account and region.
2. THE Admin_UI SHALL display each endpoint in the returned list with its name.
3. WHEN the Admin_User selects an endpoint from the list, THE Admin_UI SHALL record the selected endpoint as the active management target.
4. IF the temporary AWS credentials or region are invalid, THEN THE Endpoint_Manager SHALL return a configuration error distinct from an empty list result.
5. IF the list of SageMaker endpoints is empty, THEN THE Admin_UI SHALL display a message indicating no endpoints are available.

### Requirement 16: Endpoint Status Display and Refresh

**User Story:** As an administrator, I want to see the current endpoint status, so that I know whether the booth can serve transformations.

#### Acceptance Criteria

1. WHEN an endpoint is selected, THE Endpoint_Manager SHALL call SageMaker DescribeEndpoint directly under the Admin_Role and SHALL return the current status of the selected endpoint.
2. THE Admin_UI SHALL display the current status of the selected endpoint.
3. WHEN the Admin_User taps the refresh control, THE Endpoint_Manager SHALL re-query and return the current status of the selected endpoint and THE Admin_UI SHALL update the displayed status.
4. IF the selected endpoint does not exist when status is queried, THEN THE Endpoint_Manager SHALL return a status indicating the endpoint is not deployed and SHALL report neither an active nor an in-service status for that endpoint.

### Requirement 17: Start an Endpoint

**User Story:** As an administrator, I want to start an endpoint, so that the booth can generate images.

#### Acceptance Criteria

1. WHEN the Admin_User initiates a start action for a selected endpoint that is not deployed, THE Endpoint_Manager SHALL create the SageMaker endpoint using the configured endpoint configuration by calling SageMaker CreateEndpoint directly under the Admin_Role.
2. WHILE an endpoint start action is in progress, THE Admin_UI SHALL display a status indicating the endpoint is being created.
3. IF a start action is initiated for an endpoint that is already deployed, THEN THE Endpoint_Manager SHALL reject the action and SHALL return a message indicating the endpoint already exists.
4. WHEN the endpoint creation request is accepted by SageMaker, THE Endpoint_Manager SHALL return confirmation that creation has started.

### Requirement 18: Stop an Endpoint

**User Story:** As an administrator, I want to stop an endpoint, so that the office stops incurring cost when the booth is idle.

#### Acceptance Criteria

1. WHEN the Admin_User initiates a stop action for a selected deployed endpoint, THE Endpoint_Manager SHALL delete the SageMaker endpoint by calling SageMaker DeleteEndpoint directly under the Admin_Role.
2. BEFORE deleting the endpoint, THE Admin_UI SHALL require the Admin_User to confirm the stop action.
3. WHILE an endpoint stop action is in progress, THE Admin_UI SHALL display a status indicating the endpoint is being deleted.
4. IF a stop action is initiated for an endpoint that is not deployed, THEN THE Endpoint_Manager SHALL return a message indicating there is no endpoint to delete.

### Requirement 19: Schedule Calendar Display

**User Story:** As an administrator, I want a calendar to define endpoint working hours, so that I can plan when the endpoint runs.

#### Acceptance Criteria

1. WHEN the Admin_User opens the Schedule_Calendar, THE Schedule_Calendar SHALL display a calendar of selectable days.
2. THE Schedule_Calendar SHALL visually indicate which days have defined Working_Hours.
3. WHEN the Admin_User taps a day, THE Schedule_Calendar SHALL display an editor for defining the Working_Hours of that day.

### Requirement 20: Define and Persist Working Hours

**User Story:** As an administrator, I want to save working hours for a day, so that the schedule is retained.

#### Acceptance Criteria

1. WHEN the Admin_User sets a start time and end time for a day and confirms, THE Endpoint_Manager SHALL persist the Working_Hours for that day to the Schedule_Store by calling DynamoDB directly under the Admin_Role.
2. IF the Admin_User sets an end time that is earlier than or equal to the start time, THEN THE Schedule_Calendar SHALL reject the entry and SHALL display a validation message.
3. WHEN Working_Hours for a day are persisted, THE Schedule_Calendar SHALL update the visual indicator for that day to show defined Working_Hours.
4. WHEN the Admin_User removes the Working_Hours for a day, THE Endpoint_Manager SHALL delete the Working_Hours entry for that day from the Schedule_Store and THE Schedule_Calendar SHALL update the indicator to show no defined Working_Hours.
5. WHEN the Schedule_Calendar is reopened, THE Endpoint_Manager SHALL read the persisted Working_Hours from the Schedule_Store and THE Schedule_Calendar SHALL display the previously defined Working_Hours.

### Requirement 21: Automated Endpoint Scheduler

**User Story:** As an operator, I want the endpoint to start and stop automatically according to the defined working hours, so that the office only incurs cost while the booth is intended to be running.

#### Acceptance Criteria

1. THE Scheduler_Function SHALL be invoked on a recurring cadence of approximately one invocation every 30 seconds, implemented as two Amazon EventBridge rules whose schedules are offset by 30 seconds.
2. WHEN the Scheduler_Function is invoked, THE Scheduler_Function SHALL read the current day's Working_Hours from the Schedule_Store.
3. IF the current time is within the interval from the start time inclusive to the end time exclusive AND the endpoint is not running, THEN THE Scheduler_Function SHALL create the endpoint.
4. IF the current time is outside the interval from the start time inclusive to the end time exclusive AND the endpoint is running, THEN THE Scheduler_Function SHALL delete the endpoint.
5. WHILE the desired endpoint state equals the actual endpoint state, THE Scheduler_Function SHALL leave the endpoint unchanged.
6. THE Scheduler_Function SHALL hold IAM permissions for SageMaker ListEndpoints, DescribeEndpoint, CreateEndpoint, and DeleteEndpoint and for DynamoDB read access on the Schedule_Store.

### Requirement 22: CDK Infrastructure Provisioning

**User Story:** As an operator, I want all infrastructure deployed via CDK, so that the system is reproducible and version-controlled.

#### Acceptance Criteria

1. THE Infrastructure_Stack SHALL define all cloud resources required by the system as AWS CDK constructs.
2. THE Infrastructure_Stack SHALL provision the Auth_Service Cognito user pool and app client, the Identity_Pool, the Authenticated_Role and the Admin_Role with least-privilege policies mapped by Cognito group, the Schedule_Store DynamoDB table, the UI_Bucket, the Web_Distribution using Amazon CloudFront with Origin Access Control, the asynchronous input and output S3 bucket or references to it, the Scheduler_Function Lambda, and the two Amazon EventBridge rules.
3. THE Infrastructure_Stack SHALL exclude an Amazon API Gateway and any per-request backend Lambda functions from the provisioned resources.
4. THE Infrastructure_Stack SHALL assign a DELETE removal policy to every provisioned resource.

### Requirement 23: Initial User Parameters

**User Story:** As an operator, I want to specify initial users at deploy time, so that the system has a working admin and standard account on first launch.

#### Acceptance Criteria

1. THE Infrastructure_Stack SHALL expose a deployment parameter for the initial Admin_User username.
2. THE Infrastructure_Stack SHALL expose a deployment parameter for the initial Standard_User username.
3. WHEN the Infrastructure_Stack is deployed, THE Infrastructure_Stack SHALL create the initial Admin_User in the Auth_Service with the Profile_Attribute set to `ADMIN`.
4. WHEN the Infrastructure_Stack is deployed, THE Infrastructure_Stack SHALL create the initial Standard_User in the Auth_Service without the Profile_Attribute set to `ADMIN`.

### Requirement 24: UI Hosting

**User Story:** As an operator, I want the React UI served securely, so that visitors load the booth from a managed distribution.

#### Acceptance Criteria

1. THE Infrastructure_Stack SHALL store the built React UI assets in the UI_Bucket as a private bucket with public access blocked.
2. THE Web_Distribution SHALL serve the React UI assets from the UI_Bucket to clients.
3. THE Infrastructure_Stack SHALL restrict direct access to the UI_Bucket so that UI assets are reachable only through the Web_Distribution.
4. WHEN a client requests an application route from the Web_Distribution, THE Web_Distribution SHALL return the React application entry document so that client-side routing functions.

### Requirement 25: Portrait-Mode Touch UI Constraints

**User Story:** As a visitor, I want a clean touch-friendly interface, so that I can use the booth easily on a portrait touch screen.

#### Acceptance Criteria

1. THE Photo_Booth_App SHALL lay out every screen for portrait orientation where height exceeds width.
2. THE Photo_Booth_App SHALL render interactive controls with a minimum touch target size of 44 by 44 CSS pixels.
3. THE Photo_Booth_App SHALL limit each screen of the capture flow to the controls required for that step.
4. THE Photo_Booth_App SHALL operate using touch input without requiring a physical keyboard or mouse for the capture and effect flow.
