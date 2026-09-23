import zipfile, glob, os, re

TRAIN = '/mnt/d/flbeat/data/train'
os.makedirs(TRAIN, exist_ok=True)

# existing files (lowercased, no ext) to dedup against
existing = set(os.path.splitext(f)[0].lower() for f in os.listdir(TRAIN) if f.lower().endswith('.mp3'))
print('already have', len(existing), 'mp3s in train')

def sanitize(name):
    base = os.path.basename(name)                 # flatten any folder structure
    base = re.sub(r'[\\/:*?"<>|]', '_', base)      # NTFS-illegal chars
    return base

zips = sorted(glob.glob('/mnt/c/Users/boyan.iliev/Downloads/SpotiDownloader.com - Top 100*.zip'))
added = skipped = 0
for zp in zips:
    z = zipfile.ZipFile(zp)
    for n in z.namelist():
        if not n.lower().endswith('.mp3'):
            continue
        safe = sanitize(n)
        key = os.path.splitext(safe)[0].lower()
        if key in existing:
            skipped += 1
            continue
        dest = os.path.join(TRAIN, safe)
        with z.open(n) as src, open(dest, 'wb') as out:
            out.write(src.read())
        existing.add(key)
        added += 1
        print('  +', safe)

total = len([f for f in os.listdir(TRAIN) if f.lower().endswith('.mp3')])
print(f'\nadded {added}, skipped {skipped} dupes. train now has {total} mp3s')
