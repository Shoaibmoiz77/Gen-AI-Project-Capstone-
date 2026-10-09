# Incident Response Runbook

This runbook describes how Northwind Robotics responds to production incidents affecting customers or the robot fleet.

## Severity Levels

- **SEV1**: A full outage of the cloud dashboard, robots stopped across multiple customers, or any safety event involving a robot. Page the on-call engineer immediately.
- **SEV2**: A major feature is degraded for one or more customers, with a workaround available.
- **SEV3**: A minor issue with limited customer impact.

## Response Targets

For SEV1 incidents, the on-call engineer must acknowledge the page within 5 minutes and an incident commander must be appointed within 15 minutes. SEV2 incidents must be acknowledged within 30 minutes. SEV3 issues are handled during business hours.

## Roles

The incident commander coordinates the response and makes decisions but does not debug. A separate communications lead posts status updates. For SEV1, customer-facing status updates are posted at least every 30 minutes until resolution.

## Tooling

Pages are sent through PagerTree. Each incident gets its own Slack channel named #inc-YYYYMMDD-short-name. The status page is updated by the communications lead.

## Safety Events

If a robot is involved in any collision with a person, or a near miss reported by a customer, the affected robot fleet at that site is placed in safe mode remotely and the Head of Safety is notified within 1 hour, regardless of the time of day.

## Postmortems

A blameless postmortem is required for every SEV1 and SEV2 incident. The postmortem draft is due within 5 business days of resolution and is reviewed at the weekly reliability meeting. Action items are tracked as tickets with an owner and due date.

## On-call

Engineers rotate on-call weekly, with handoff on Mondays at 10:00 local time. On-call engineers receive a stipend of $400 per week of on-call duty.
