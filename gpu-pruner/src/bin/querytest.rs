/// Scratch testing/debug script for querying Prometheus and checking outputs
use chrono::{DateTime, NaiveDateTime, Utc};
use gpu_pruner::*;
use prettytable::{Cell, Row, Table};
use prometheus_http_query::response::Data;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    // get first arg
    let args: Vec<String> = std::env::args().collect();

    let query = args.get(1).expect("Query is required");
    let prometheus_url: &str = args.get(2).expect("Prometheus URL is required");

    let prometheus_tls_cert: Option<&str> = Some("certs/prometheus.crt");

    tracing::info!("Prometheus URL: {}", &prometheus_url);
    tracing::info!("Query: {}", &query);

    let token = get_prometheus_token().await?;
    let client = get_prom_client(prometheus_url, token, TlsMode::Verify, prometheus_tls_cert)?;

    let data = client.query(query).get().await?;

    let mut table = Table::new();

    match data.data() {
        Data::Vector(sm) => {
            for (i, result) in sm.iter().enumerate() {
                let mut cells: Vec<Cell> = result
                    .metric()
                    .values()
                    .into_iter()
                    .map(|x| Cell::new(x))
                    .collect();

                cells.push(Cell::new(&convert_timestamp_to_string(
                    result.sample().timestamp(),
                )));
                cells.push(Cell::new(&result.sample().value().to_string()));

                table.add_row(Row::new(cells));
            }
        }
        Data::Matrix(sm) => {
            for result in sm {
                for (i, sample) in result.samples().iter().enumerate() {
                    let mut cells: Vec<Cell> = result
                        .metric()
                        .values()
                        .into_iter()
                        .map(|x| Cell::new(x))
                        .collect();

                    cells.push(Cell::new(&convert_timestamp_to_string(sample.timestamp())));
                    cells.push(Cell::new(&sample.value().to_string()));

                    table.add_row(Row::new(cells));
                }
            }
        }
        Data::Scalar(sm) => {
            tracing::error!("Scalar data not supported");
        }
    }

    table.printstd();
    let mut file = std::fs::File::create("output.csv")?;
    table.to_csv(&mut file)?;

    Ok(())
}

fn convert_timestamp_to_string(timestamp: f64) -> String {
    let naive = NaiveDateTime::from_timestamp(timestamp as i64, 0);
    let datetime: DateTime<Utc> = DateTime::from_utc(naive, Utc);
    datetime.to_string()
}
