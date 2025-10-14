import Link from 'next/link'
import Hero from '@/components/Hero'
import { StatCard } from '@/components/StatCard'

export default function Page() {
  return (
    <>
      <Hero
        title="Data Valuation — BMW × Clemson Midterm Update"
        subtitle="Four workstreams: Cost, Benefit, Integrated Model, and Alternative (Hedonic). This site summarizes what we’ve done and what’s next."
        cta={<Link href="/integrated" className="inline-block bg-brand hover:bg-brand-dark text-black font-semibold px-5 py-3 rounded-lg">See Combined View</Link>}
      />
      <section className="grid md:grid-cols-3 gap-6">
        <StatCard label="Monte Carlo Runs" value="5,000+" foot="Cost uncertainty modeled as ranges" />
        <StatCard label="Benefit Channels" value="7" foot="OEE, downtime, quality, safety, energy..." />
        <StatCard label="Comps Collected" value="20+" foot="For hedonic modeling (growing)" />
      </section>

      <div className="grid md:grid-cols-2 gap-6 mt-10">
        <div className="card p-6">
          <h3 className="font-semibold text-lg">Cost Team</h3>
          <p className="text-white/70 mt-2">Ranges → Monte Carlo → P50/P90 + Tornado drivers.</p>
          <Link href="/cost" className="inline-block mt-4 text-brand">Open →</Link>
        </div>
        <div className="card p-6">
          <h3 className="font-semibold text-lg">Benefit Team</h3>
          <p className="text-white/70 mt-2">Throughput, quality, downtime, safety, energy, risk.</p>
          <Link href="/benefit" className="inline-block mt-4 text-brand">Open →</Link>
        </div>
        <div className="card p-6">
          <h3 className="font-semibold text-lg">Integrated</h3>
          <p className="text-white/70 mt-2">Unified view: value – cost; packaging & price bands.</p>
          <Link href="/integrated" className="inline-block mt-4 text-brand">Open →</Link>
        </div>
        <div className="card p-6">
          <h3 className="font-semibold text-lg">Alternative (Hedonic)</h3>
          <p className="text-white/70 mt-2">Market comps: rights × freshness × coverage → price.</p>
          <Link href="/alternative" className="inline-block mt-4 text-brand">Open →</Link>
        </div>
      </div>
    </>
  )
}
