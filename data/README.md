`names.txt` is a componsed of names from,
* https://en.wikipedia.org/wiki/List_of_unicorn_startup_companies
* https://www.sec.gov/files/rules/other/4-460list.htm
* https://www.sec.gov/divisions/corpfin/internatl/alphabetical.htm
with corporate suffixes (inc., co., group, etc.) removed.

`sec__edgar_company_names.txt` is generated from
`raw/sec__edgar_company_info.csv`
(downloaded from
https://www.kaggle.com/datasets/dattapiy/sec-edgar-companies-list) using
`notebook/preprocess_sec_edgar_company_names.ipynb`.

`odbus_v1.csv` is generated from `raw/ODBus_v1/ODBus_v1.csv1`.

`startup_investments_crunchbase.csv` is generated from
`raw/startup_investments_crunchbase.csv` (downloaded from
https://www.kaggle.com/datasets/arindam235/startup-investments-crunchbase).

`raw/unicorns_till_sep_2022.csv` was downloaded from
https://www.kaggle.com/datasets/ramjasmaurya/unicorn-startups.

`raw/gd_auctions_export_1715053158020.csv` was downloaded from
https://auctions.godaddy.com/ on May 6, 2024.