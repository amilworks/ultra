Sample fixture tree for local zero-copy NFS registration tests.

This directory is mounted read-only at /mnt/bisque_legacy_data by
docker-compose.nfs-local.yml so `bq-path sync` can register files in place
without copying their bytes into BisQue's managed upload area.
