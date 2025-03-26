# Freebase Setup

## Requirements

- OpenLink Virtuoso 7.2.5 (download from this public [link](https://sourceforge.net/projects/virtuoso/files/virtuoso/))
```bash
wget https://sourceforge.net/projects/virtuoso/files/virtuoso/7.2.5/virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz
```
- Python 3
- Freebase dump from this public [link](https://developers.google.com/freebase?hl=en)

## Setup

### Data Preprocessing

We use this py script (public [link)](https://github.com/lanyunshi/Multi-hopComplexKBQA/blob/master/code/FreebaseTool/FilterEnglishTriplets.py), to clean the data and remove non-English or non-digital triplets:

```shell
gunzip -c freebase-rdf-latest.gz > freebase # data size: 400G
nohup python -u FilterEnglishTriplets.py 0<freebase 1>FilterFreebase 2>log_err & # data size: 125G
```

## Import data

we import the cleaned data to virtuoso, 

```shell
tar xvpfz virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz
cd virtuoso-opensource/database/
mv virtuoso.ini.sample virtuoso.ini

# ../bin/virtuoso-t -df # start the service in the shell
../bin/virtuoso-t  # start the service in the backend.
../bin/isql 2111 dba dba # run the database

# 1、unzip the data and use rdf_loader to import
SQL>
ld_dir('/data1/zhuom/', 'FilterFreebase', 'http://freebase.com'); 
rdf_loader_run(); 

# close virtuoso
cd /data1/zhuom/virtuoso-opensource/database/
killall virtuoso-t   # Ensure the old instance stops
```

Wait for a long time and then ready to use.

Check how many triplets loaded:
```sql
SELECT COUNT(*) FROM DB.DBA.RDF_QUAD;
```

```

## 
