import Hero from '@/components/Hero'
import Section from '@/components/Section'
import Gallery from '@/components/Gallery'
import Link from 'next/link'

export default function AlternativePage() {
  return (
    <>
      <Hero title="Alternative — Hedonic Market Approach" subtitle="Use market comps to learn attribute price premiums (rights, freshness/latency, coverage, history)." />
      <Section title="Current Visuals">
        <Gallery images={[
          { src: '/uploads/price_ladder.png', caption: 'Illustrative Price Ladder (rights × freshness)' },
          { src: '/uploads/attribute_card.png', caption: 'BMW Attribute Card (draft)' }
        ]} />
      </Section>
      <Section title="Leak Library & Relevance Scoring">
        <p className="text-white/80">We’re collecting auto/industrial leakage cases to calibrate expected loss and tail risk, then folding that into our security spend curve.</p>
        <div className="mt-4">
          <Link href="/uploads/lawsuit_data_leakages_scored.csv" className="text-brand underline">Download: Scored Cases CSV</Link>
        </div>
      </Section>
    </>
  )
}
