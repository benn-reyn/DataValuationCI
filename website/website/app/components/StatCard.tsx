export function StatCard({ label, value, foot }: { label: string; value: string; foot?: string }) {
  return (
    <div className="card p-5">
      <div className="text-white/60 text-sm">{label}</div>
      <div className="text-2xl font-bold mt-2">{value}</div>
      {foot && <div className="text-white/50 text-xs mt-2">{foot}</div>}
    </div>
  )
}
