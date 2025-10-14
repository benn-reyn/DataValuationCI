import Hero from '@/components/Hero'
import Section from '@/components/Section'
import Gallery from '@/components/Gallery'

export default function CostPage() {
  return (
    <>
      <Hero title="Cost — Ranges & Monte Carlo" subtitle="We moved from point assumptions to min / likely / max and simulated outcomes to get P50/P90 and driver sensitivity." />
      <Section title="Assumption Ranges (editable)">
        <ul className="list-disc ml-6 text-white/85 space-y-1">
          <li>Storage (cTB): 18 → <b>23</b> → 30 $/TB-month</li>
          <li>Labeling (w×τ×φ): τ by task complexity; φ for QA & tools</li>
          <li>Ops (α, β): processing per TB and per-model MLOps</li>
          <li>Risk: baseline probability, security half-life, expected loss</li>
        </ul>
      </Section>
      <Section title="Simulation Outputs">
        <Gallery images={[
          { src: '/uploads/mc_histogram.png', caption: 'Distribution of monthly cost (P50 & P90 markers)' },
          { src: '/uploads/tornado.png', caption: 'Tornado: cost drivers ranked by impact' }
        ]} />
      </Section>
    </>
  )
}
