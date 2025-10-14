export default function Hero({
  title, subtitle, cta
}: { title: string; subtitle?: string; cta?: React.ReactNode }) {
  return (
    <section className="mb-10">
      <div className="card p-8 md:p-12">
        <h1 className="text-3xl md:text-5xl font-bold leading-tight">{title}</h1>
        {subtitle && <p className="mt-4 text-white/80 md:text-lg">{subtitle}</p>}
        {cta && <div className="mt-6">{cta}</div>}
      </div>
    </section>
  )
}
