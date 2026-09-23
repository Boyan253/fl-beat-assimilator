import zipfile, glob
zips = glob.glob('/mnt/c/Users/boyan.iliev/Downloads/SpotiDownloader.com - Top 100*.zip')
print('found zips:', len(zips))
for f in zips:
    print('===', f.split('/')[-1], '===')
    try:
        z = zipfile.ZipFile(f)
        names = z.namelist()
        mp3 = [n for n in names if n.lower().endswith('.mp3')]
        print('  entries:', len(names), ' mp3:', len(mp3))
        for n in mp3[:6]:
            print('   ', n)
    except Exception as e:
        print('  ERROR:', type(e).__name__, e)
