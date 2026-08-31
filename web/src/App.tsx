import "@patternfly/react-core/dist/styles/base.css";
import {
  Page,
  PageSection,
  PageSectionVariants,
  Title,
} from "@patternfly/react-core";
import { IdleGpuLeaderboard } from "./components/IdleGpuLeaderboard";
import { StatsDashboard } from "./components/StatsDashboard";

export default function App() {
  return (
    <Page>
      <PageSection variant={PageSectionVariants.secondary}>
        <Title headingLevel="h1" size="2xl">
          GPU Pruner
        </Title>
        <p style={{ marginTop: "0.5rem", color: "#6a6e73" }}>
          Monitor idle GPU workloads and scale-down activity.
        </p>
      </PageSection>
      <PageSection>
        <StatsDashboard />
        <IdleGpuLeaderboard />
      </PageSection>
    </Page>
  );
}
