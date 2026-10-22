# The live engine on AWS

`qfin live` runs one session of the QQQ strategy on Alpaca's paper account (decision
0021). On AWS it is an ECS Fargate task, `qfin-live`, beside the nightly `qfin-alphaguard`
job: same cluster, image, roles and log group.

| | |
| --- | --- |
| When | every weekday at 09:15 New York time (EventBridge Scheduler, `America/New_York`, so the clocks' changes need nothing); on a holiday it says "no session" and ends |
| Ends | about two minutes after the close, once the day's orders have expired |
| Size | 0.25 vCPU, 1 GB: about 2 USD a month |
| Journal | the task's disk lasts as long as the task, so the day's folder goes to `s3://qfin-alphaguard-kefebekir/data/live/<day>/` every minute and at the end; a task started again that day fetches it first and replays it |
| Alerts | the SNS topic `qfin-alerts`, e-mailed to Efe; a summary when a session closes |
| Report | `qfin-live-report`, each weekday at 16:30 New York time, once the consolidated tape serves the day's last minutes: the session against backtests of it (`qfin live-report`), its summary e-mailed, `report.txt` and `report.html` kept in the day's folder |
| Keys | Alpaca's paper keys in SSM, `/qfin/alpaca-key-id` and `/qfin/alpaca-secret-key` |
| News capture | `qfin-news-capture`, each weekday at 15:31 New York time: the news as it stands then, for N5's shadow record (`qfin news-capture`), to `s3://qfin-alphaguard-kefebekir/data/news_capture/<day>.jsonl`, once a session and never written again; it runs only between 15:30 and 16:30 New York time, and a failure is e-mailed |

Do not run `qfin live --check` while a session runs: Alpaca's free plan allows one market
data connection, and the check's would be refused.

## Setting it up

Every command takes `--profile qfin --region eu-north-1`; `ACCOUNT_ID` is the account's
number.

1. **The keys** (Efe, from the repository, so they are never printed):

   ```bash
   set -a && source <(grep -E "^ALPACA_API_(KEY_ID|SECRET_KEY)=" .env) && set +a
   aws ssm put-parameter --name /qfin/alpaca-key-id --type SecureString --value "$ALPACA_API_KEY_ID" --overwrite
   aws ssm put-parameter --name /qfin/alpaca-secret-key --type SecureString --value "$ALPACA_API_SECRET_KEY" --overwrite
   ```

2. **The alerts**: `aws sns create-topic --name qfin-alerts`, then
   `aws sns subscribe --topic-arn <its ARN> --protocol email --notification-endpoint <Efe's e-mail>`;
   Efe confirms from the e-mail AWS sends.

3. **The roles**, from this folder's policies:
   - `aws iam put-role-policy --role-name qfin-task-role --policy-name write-prices-to-s3 --policy-document file://infra/aws/task-role-policy.json`
   - `aws iam put-role-policy --role-name qfin-execution-role --policy-name qfin-read-eodhd-key --policy-document file://infra/aws/execution-role-secrets-policy.json`
   - `aws iam put-role-policy --role-name qfin-scheduler-role --policy-name run-qfin-task --policy-document file://infra/aws/scheduler-role-policy.json`

4. **The image**, from `main`, with Docker Desktop open: build, then push as `:latest`
   and `:<short commit>`, as for the nightly job. The nightly job runs `:latest` too.

5. **The tasks**: `sed s/ACCOUNT_ID/<number>/g` on `live-task-definition.template.json` and
   `live-report-task-definition.template.json`, each into
   `aws ecs register-task-definition --cli-input-json file://...`.

6. **A check**: `aws ecs run-task --cluster qfin --task-definition qfin-live --launch-type FARGATE`,
   with the nightly schedule's network configuration and the command overridden to
   `["live", "--check"]`; its log is in `/ecs/qfin-alphaguard`, stream `live/qfin-live/<task>`.

7. **The news capture** (N5): with the scheduler role's policy put again from
   `scheduler-role-policy.json` (it names `qfin-news-capture`), `sed` on
   `news-capture-task-definition.template.json` into `aws ecs register-task-definition`, then
   `aws scheduler create-schedule --name qfin-news-capture` with `cron(31 15 ? * MON-FRI *)`
   in `America/New_York`, a flexible window of `OFF`, and the same role and network
   configuration as the others.

8. **The schedules**: `aws scheduler create-schedule --name qfin-live`
   with `cron(15 9 ? * MON-FRI *)` in `America/New_York`, a flexible window of `OFF`, and as
   its target `qfin-live`'s task definition, with the nightly schedule's role and network
   configuration; and `qfin-live-report` the same way, with `cron(30 16 ? * MON-FRI *)` and
   `qfin-live-report`'s task definition.

A change to `qfin live`'s command line needs a new revision of `qfin-live` and the
schedule pointed at it, as for the nightly job.
