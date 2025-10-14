export default function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mb-10">
      <h2 className="text-xl md:text-2xl font-semibold mb-4">{title}</h2>
      <div className="card p-6 md:p-8">{children}</div>
    </section>
  )
}
