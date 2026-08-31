use anyhow::Result;
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde_json::json;
use sha2::Sha256;

use crate::Meta;

/// Maximum allowed skew for the X-Slack-Request-Timestamp header.
const MAX_TIMESTAMP_SKEW_SECS: i64 = 300;

#[derive(Clone, Debug)]
pub struct SlackNotifier {
    webhook_url: String,
    client: Client,
    channel: Option<String>,
}

impl SlackNotifier {
    pub fn new(webhook_url: String, channel: Option<String>) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()?;

        Ok(Self {
            webhook_url,
            client,
            channel,
        })
    }

    #[tracing::instrument(skip(self, workload))]
    pub async fn send_notification<T: Meta + std::fmt::Debug>(
        &self,
        workload: &T,
        idle_duration_minutes: i64,
        ack_grace_period_secs: u64,
        mentions: Option<String>,
    ) -> Result<()> {
        let resource_type = workload.kind();
        let resource_name = workload.name();
        let namespace = workload
            .namespace()
            .unwrap_or_else(|| "default".to_string());

        let grace_minutes = ack_grace_period_secs / 60;
        let grace_label = if ack_grace_period_secs.is_multiple_of(60) && grace_minutes > 0 {
            format!("{grace_minutes} minutes")
        } else {
            format!("{ack_grace_period_secs} seconds")
        };

        let button_value =
            |hours: u32| format!("{resource_type}:{namespace}:{resource_name}:{hours}");

        let mut payload = json!({
            "attachments": [{
                "callback_id": "ack_idle_gpu",
                "color": "warning",
                "title": "Idle GPU Detected",
                "fields": [
                    {
                        "title": "Resource",
                        "value": format!("{}: {}", resource_type, resource_name),
                        "short": true
                    },
                    {
                        "title": "Namespace",
                        "value": namespace,
                        "short": true
                    },
                    {
                        "title": "Reason",
                        "value": format!("GPU idle for {} minutes", idle_duration_minutes),
                        "short": false
                    },
                    {
                        "title": "Action",
                        "value": format!("You have {grace_label} to acknowledge before scale-down"),
                        "short": false
                    }
                ],
                "actions": [
                    {
                        "name": "ack",
                        "text": "Keep 4h",
                        "type": "button",
                        "value": button_value(4),
                        "style": "primary"
                    },
                    {
                        "name": "ack",
                        "text": "Keep 8h",
                        "type": "button",
                        "value": button_value(8),
                        "style": "primary"
                    },
                    {
                        "name": "ack",
                        "text": "Keep 24h",
                        "type": "button",
                        "value": button_value(24),
                        "style": "primary"
                    }
                ],
                "footer": "gpu-pruner",
                "ts": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            }]
        });

        if let Some(channel) = &self.channel {
            payload["channel"] = json!(channel);
        }
        if let Some(mention_text) = mentions {
            payload["text"] = json!(mention_text);
        }

        tracing::debug!("Sending Slack notification payload: {:?}", payload);

        let response = self
            .client
            .post(&self.webhook_url)
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            tracing::error!(
                status = %status,
                body = %body,
                "Slack webhook returned error status"
            );
            return Err(anyhow::anyhow!(
                "Slack webhook failed with status {}: {}",
                status,
                body
            ));
        }

        tracing::info!("Sent Slack notification for [{resource_type}] {namespace}:{resource_name}",);

        Ok(())
    }

    /// POST a JSON message to a Slack `response_url`.
    pub async fn post_response(
        &self,
        response_url: &str,
        message: &serde_json::Value,
    ) -> Result<()> {
        let response = self.client.post(response_url).json(message).send().await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Slack response_url post failed with status {}",
                response.status()
            ));
        }
        Ok(())
    }
}

/// Verify a Slack request signature (`v0` scheme).
///
/// <https://api.slack.com/authentication/verifying-requests-from-slack>
pub fn verify_slack_signature(
    signing_secret: &str,
    timestamp: &str,
    body: &str,
    signature: &str,
) -> bool {
    verify_slack_signature_at(
        signing_secret,
        timestamp,
        body,
        signature,
        chrono::Utc::now().timestamp(),
    )
}

fn verify_slack_signature_at(
    signing_secret: &str,
    timestamp: &str,
    body: &str,
    signature: &str,
    now_epoch_secs: i64,
) -> bool {
    let Ok(ts) = timestamp.parse::<i64>() else {
        return false;
    };
    if (now_epoch_secs - ts).abs() > MAX_TIMESTAMP_SKEW_SECS {
        return false;
    }

    let Some(hex_sig) = signature.strip_prefix("v0=") else {
        return false;
    };
    let Ok(expected) = hex::decode(hex_sig) else {
        return false;
    };

    let Ok(mut mac) = Hmac::<Sha256>::new_from_slice(signing_secret.as_bytes()) else {
        return false;
    };
    mac.update(format!("v0:{timestamp}:{body}").as_bytes());
    mac.verify_slice(&expected).is_ok()
}

#[cfg(test)]
mod tests {
    use crate::slack::{SlackNotifier, verify_slack_signature_at};

    // Worked example from the Slack request-verification docs.
    const DOC_SECRET: &str = "8f742231b10e8888abcd99yyyzzz85a5";
    const DOC_TIMESTAMP: &str = "1531420618";
    const DOC_BODY: &str = "token=xyzz0WbapA4vBCDEFasx0q6G&team_id=T1DC2JH3J&team_domain=testteamnow&channel_id=G8PSS9T3V&channel_name=foobar&user_id=U2CERLKJA&user_name=roadrunner&command=%2Fwebhook-collect&text=&response_url=https%3A%2F%2Fhooks.slack.com%2Fcommands%2FT1DC2JH3J%2F397700885554%2F96rGlfmibIGlgcZRskXaIFfN&trigger_id=398738663015.47445629121.803a0bc887a14d10d2c447fce8b6703c";
    const DOC_SIGNATURE: &str =
        "v0=a2114d57b48eac39b9ad189dd8316235a7b4a8d21a10bd27519666489c69b503";

    #[test]
    fn signature_matches_slack_docs_example() {
        let ts: i64 = DOC_TIMESTAMP.parse().unwrap();
        assert!(verify_slack_signature_at(
            DOC_SECRET,
            DOC_TIMESTAMP,
            DOC_BODY,
            DOC_SIGNATURE,
            ts + 10,
        ));
    }

    #[test]
    fn signature_rejects_tampered_body() {
        let ts: i64 = DOC_TIMESTAMP.parse().unwrap();
        assert!(!verify_slack_signature_at(
            DOC_SECRET,
            DOC_TIMESTAMP,
            &format!("{DOC_BODY}x"),
            DOC_SIGNATURE,
            ts + 10,
        ));
    }

    #[test]
    fn signature_rejects_wrong_secret() {
        let ts: i64 = DOC_TIMESTAMP.parse().unwrap();
        assert!(!verify_slack_signature_at(
            "not-the-secret",
            DOC_TIMESTAMP,
            DOC_BODY,
            DOC_SIGNATURE,
            ts + 10,
        ));
    }

    #[test]
    fn signature_rejects_stale_timestamp() {
        let ts: i64 = DOC_TIMESTAMP.parse().unwrap();
        assert!(!verify_slack_signature_at(
            DOC_SECRET,
            DOC_TIMESTAMP,
            DOC_BODY,
            DOC_SIGNATURE,
            ts + 301,
        ));
    }

    #[test]
    fn signature_rejects_malformed_inputs() {
        assert!(!verify_slack_signature_at(
            DOC_SECRET,
            "not-a-number",
            DOC_BODY,
            DOC_SIGNATURE,
            0
        ));
        let ts: i64 = DOC_TIMESTAMP.parse().unwrap();
        assert!(!verify_slack_signature_at(
            DOC_SECRET,
            DOC_TIMESTAMP,
            DOC_BODY,
            "missing-prefix",
            ts,
        ));
        assert!(!verify_slack_signature_at(
            DOC_SECRET,
            DOC_TIMESTAMP,
            DOC_BODY,
            "v0=nothex",
            ts,
        ));
    }

    #[test]
    fn notifier_builds_without_channel() {
        assert!(SlackNotifier::new("https://hooks.slack.com/services/TEST".into(), None).is_ok());
    }
}
