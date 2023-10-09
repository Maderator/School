# MetaCentrum

## Frontend
Přihlaš se na frontend. Já např. používám `skirit.ics.muni.cz` na FI MUNI v Brně, cítím se tam doma. ;-)
```
ssh petrkutalek@skirit.ics.muni.cz
```

Na tento stroj také přenášíš data, pokud potřebuješ něco pro svou úlohu, nebo si chceš stáhnout výsledky:
```
scp petrkutalek@skirit.ics.muni.cz:data.tar /home/petr/Desktop/
```

Data si můžeš umístit na nějaký home v dané lokalitě, kde využíváš frontend, třeba `/storage/brno2/home/petrkutalek/…`.

Zde si nyní požádáš o *interaktivní* session na backend, zde na 4 hodiny:
```
qsub -I -l select=1:mem=4gb:scratch_local=1gb:ngpus=1:gpu_cap=cuda61:cuda_version=11.0 -q gpu -l walltime=4:00:00
```

## Backend Interaktivně

Počkáš na přidělení a následně můžeš pracovat!

```
mkdir $SCRATCHDIR/tmp
export SINGULARITY_TMPDIR=$SCRATCHDIR/tmp
ls /cvmfs/singularity.metacentrum.cz
singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/TensorFlow\:21.02-tf2-py3.SIF

...
python -c  'import torch; print(torch.cuda.get_device_properties(0))'
```

## Backend přes qsub
...
