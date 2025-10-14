import Hero from '@/components/Hero'
import Section from '@/components/Section'
import Gallery from '@/components/Gallery'

export default function BenefitPage() {
  return (
    <>
      <Hero title="Benefit — Value Channels" subtitle="Throughput (OEE), quality, downtime, safety, energy, and risk reduction." />
      <Section title="Key Findings">
        <ul className="list-disc ml-6 text-white/85 space-y-1">
          <li>OEE uplift + downtime reduction dominate value</li>
          <li>Quality & safety add meaningful secondary gains</li>
          <li>Risk reduction sized with realistic λ</li>
        </ul>
      </Section>
      <Section title="Visuals">
        <Gallery images={[
          { src: '/uploads/benefit_bars.png', caption: 'Expected annual benefits by channel' },
          { src: '/uploads/before_after.png', caption: 'Before/After: OEE, Defects, Downtime, Variability' }
        ]} />
      </Section>
    </>
  )
}
