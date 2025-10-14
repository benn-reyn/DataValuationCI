import Hero from '@/components/Hero'
import Section from '@/components/Section'
import Image from 'next/image'

export default function IntegratedPage() {
  return (
    <>
      <Hero title="Integrated — Value minus Cost" subtitle="One model that connects operational benefits and cost; guides packaging and terms." />
      <Section title="Concept (Preview)">
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold">Unified Equation (sketch)</h3>
            <p className="text-white/80 mt-2">Net Value = f(OEE, quality, downtime, safety, energy, risk) – g(storage, labeling, ops, security).</p>
            <p className="text-white/60 mt-3 text-sm">Planned: sensitivity to freshness tiers (daily/hourly/real-time) and license scope (internal/redistribution) → a <b>price ladder</b>.</p>
          </div>
          <div className="card p-4">
            <Image src="/uploads/integrated_sankey.png" width={1600} height={900} alt="Integrated flow" className="w-full h-auto rounded-md" />
          </div>
        </div>
      </Section>
    </>
  )
}
