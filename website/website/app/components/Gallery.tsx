import Image from 'next/image'

export default function Gallery({ images }: { images: { src: string; caption?: string }[] }) {
  return (
    <div className="grid md:grid-cols-2 gap-6">
      {images.map((im, i) => (
        <figure key={i} className="card overflow-hidden">
          <Image src={im.src} alt={im.caption || `image-${i}`} width={1600} height={900} className="w-full h-auto" />
          {im.caption && <figcaption className="p-3 text-white/70 text-sm">{im.caption}</figcaption>}
        </figure>
      ))}
    </div>
  )
}
