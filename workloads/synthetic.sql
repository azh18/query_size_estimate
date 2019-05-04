SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year>2012
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<475 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=1 AND t.kind_id<7 AND t.production_year<2010
SELECT COUNT(*) FROM title t WHERE t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id<8 AND t.kind_id=7 AND t.production_year>1950
SELECT COUNT(*) FROM movie_info_idx mi_idx WHERE mi_idx.info_type_id>99
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>5804
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=21038
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=261
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=4
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>46
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id>44813 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx WHERE mi_idx.info_type_id<101
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>4609 AND ci.person_id>132552 AND t.kind_id<7 AND t.production_year>2010
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=6 AND t.kind_id=3 AND t.production_year=2007
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=78884
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<13529 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id<7 AND t.production_year=2013
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>3235 AND t.kind_id=1 AND t.production_year=2005
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND t.kind_id=1 AND t.production_year>1931
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=2561 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<7 AND t.production_year<2012
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1016550
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<3006
SELECT COUNT(*) FROM movie_info_idx mi_idx WHERE mi_idx.info_type_id=100
SELECT COUNT(*) FROM title t WHERE t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<498581 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>45241
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id<18 AND mi_idx.info_type_id=99
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id<7 AND t.production_year>2012
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id>144283 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=100
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<992971 AND mc.company_id>2176 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND ci.person_id>1289897 AND ci.role_id<10 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<16 AND ci.person_id>3401709 AND t.kind_id=7 AND t.production_year<1995
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>3040879 AND mi_idx.info_type_id>101
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>2004
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id<7 AND t.production_year=2000
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>105 AND mk.keyword_id=1198 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3117567 AND t.kind_id>1 AND t.production_year>2011
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id>99 AND t.kind_id>1 AND t.production_year>2011
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=15927
SELECT COUNT(*) FROM movie_info_idx mi_idx WHERE mi_idx.info_type_id>100
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16 AND ci.person_id<2871027 AND ci.role_id>1 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND mc.company_id=100874 AND t.kind_id<7 AND t.production_year>1957
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>4 AND t.production_year<1962
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>37865 AND t.kind_id=2
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id>10038
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>947 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=3289
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id=335 AND t.kind_id>3 AND t.production_year=2013
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<3929741 AND t.kind_id>1 AND t.production_year>2009
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>498 AND ci.role_id>10
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<106
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year=1980
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=5 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id<75402 AND ci.role_id>2 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx WHERE mi_idx.info_type_id>101
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=3 AND t.kind_id<7
SELECT COUNT(*) FROM title t WHERE t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>3250 AND mc.company_type_id<2 AND t.kind_id=7 AND t.production_year>2012
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id=1 AND t.production_year=2008
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>2016565 AND ci.role_id=3 AND t.kind_id=7 AND t.production_year<1978
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=680 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>73674
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id=2317194
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>2546
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=92
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=99 AND t.kind_id=7 AND t.production_year>1910
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id=1021637
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>39735
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>4465
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND mk.keyword_id>25936 AND t.kind_id=1 AND t.production_year>2007
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<312693 AND ci.role_id<3 AND mi_idx.info_type_id>99 AND t.kind_id<2
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>99
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND mk.keyword_id<2396
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>1631472 AND ci.role_id=1 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3168324 AND ci.role_id>2
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND mc.company_id<11 AND mc.company_type_id=1 AND t.kind_id=3
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year>1926
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id=3 AND t.production_year>2004
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=67104
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>4 AND mk.keyword_id>1059 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=29670
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=99 AND t.kind_id=1 AND t.production_year=2004
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=2185
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id=75861 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=11302 AND t.kind_id>2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND t.kind_id<4
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>16
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=2860
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id>1816442 AND ci.role_id>2 AND t.kind_id>1 AND t.production_year<1989
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>2
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year=2005
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=2536314 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2461204 AND ci.role_id>2
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=7 AND t.production_year<1975
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=22230
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND mc.company_id>65820 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id<32469
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year>1992
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>951606 AND t.kind_id=1 AND t.production_year>1971
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=2017
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND mi.info_type_id=9
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=99
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id>4 AND t.production_year>1983
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<3682092 AND mk.keyword_id=12284
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2917545 AND ci.role_id>1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=1427 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1480684
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<8 AND t.kind_id<7 AND t.production_year<1985
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>750
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>6 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=317282
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id>13015 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id<7
SELECT COUNT(*) FROM title t WHERE t.kind_id>1
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>17
SELECT COUNT(*) FROM movie_info_idx mi_idx WHERE mi_idx.info_type_id=99
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>2 AND ci.person_id=3914905
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id>1381 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id<8
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=69620
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year=2010
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<37791
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<16 AND mk.keyword_id>4618 AND t.kind_id<4 AND t.production_year>2011
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1103909 AND ci.role_id<10
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>18
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=96790
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=22774
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>1967
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id=107 AND mi_idx.info_type_id<101 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id=288478
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<3141404 AND ci.role_id=1 AND mi.info_type_id>15 AND t.kind_id=7 AND t.production_year>1966
SELECT COUNT(*) FROM movie_info_idx mi_idx WHERE mi_idx.info_type_id=101
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.kind_id>1 AND t.production_year<1994
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<11203
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>101 AND t.kind_id<3 AND t.production_year<2008
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>4
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id=7637
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year=2012
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<6
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3341360
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>16
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<3897107 AND ci.role_id=4 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=1044
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year=2012
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=4 AND ci.person_id>1466320 AND t.kind_id<7 AND t.production_year<2013
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>58867 AND ci.role_id>5 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=335 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<2256517
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=5
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=14260
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id>100 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=145
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=90679
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>7552
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id>3
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>6
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>58853 AND mi.info_type_id=8 AND t.kind_id>4
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<16
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1137015 AND ci.role_id>2 AND t.kind_id>1 AND t.production_year<2011
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>100
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<29406
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<73527
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=98 AND mk.keyword_id<3086
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8 AND mk.keyword_id<11717
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1083600
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=687 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>3283
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1280109
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year>2008
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id<7 AND t.production_year<2003
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>105 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=347
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=1728643 AND ci.role_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<7 AND t.kind_id=7 AND t.production_year=1967
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id=6484
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=4925
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<73349
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>439 AND ci.person_id>3460343 AND ci.role_id>1 AND t.kind_id<3
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=16
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>196163 AND t.kind_id>1 AND t.production_year>1951
SELECT COUNT(*) FROM title t WHERE t.kind_id<3
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>3024 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>276 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id<107 AND mk.keyword_id=9038
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>16264
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1877127
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<3429307
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=3
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id<101 AND mk.keyword_id<968 AND t.kind_id<7 AND t.production_year>1992
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>1464890 AND ci.role_id>8
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=16141
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id=2770817 AND ci.role_id>1 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=98
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND mi.info_type_id=1 AND t.kind_id>1 AND t.production_year=2003
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1721883 AND ci.role_id>3
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<4226 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>101 AND mk.keyword_id<1358
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>72053 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>99 AND t.kind_id=1 AND t.production_year<1978
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>2424 AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=1640
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id<101 AND mk.keyword_id=438
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<94
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>1700794 AND mk.keyword_id<3312
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>16 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1999625 AND ci.role_id>1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<19
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>6269
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1209687 AND ci.role_id>2
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=7 AND t.production_year>2012
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=10294
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id>1 AND t.production_year<2005
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>18 AND mk.keyword_id>17174
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<5561
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<7 AND t.production_year>2010
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1783216 AND ci.role_id<7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id<23766
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>73 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1234751
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=7 AND t.production_year=1996
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<10700 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>2551
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>402
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<8
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<131723
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>16026 AND t.kind_id<7 AND t.production_year<1995
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<1661
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>14489
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year=2011
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id=16 AND t.kind_id=2 AND t.production_year>2012
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>305
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>2 AND mc.company_type_id=1 AND mi.info_type_id=98 AND t.kind_id<4 AND t.production_year>2005
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.kind_id<2 AND t.production_year>2003
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<79171 AND t.kind_id=7 AND t.production_year<2011
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<3761
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id<74784 AND mc.company_type_id=1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>1159361 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id>1 AND t.production_year>2012
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id<6 AND mk.keyword_id<2108 AND t.kind_id=2 AND t.production_year=2009
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<160 AND mc.company_type_id=2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1184432
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<3657190 AND t.kind_id=1 AND t.production_year>1984
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101 AND mk.keyword_id=9347
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id=2427308
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id=16 AND mk.keyword_id>13241
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2657721
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<662903 AND t.kind_id=7 AND t.production_year>1978
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.kind_id<7 AND t.production_year>2001
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2014171 AND ci.role_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=6158
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=17 AND t.kind_id=7 AND t.production_year>2011
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<98 AND ci.person_id<1351178 AND ci.role_id>9 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND ci.person_id>2866549 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2382212
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<18
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<972266 AND ci.role_id<10
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>761538 AND ci.role_id>1 AND t.kind_id<4 AND t.production_year>1978
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1052817
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<34036 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<3232 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2025429 AND ci.role_id<9
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101 AND mk.keyword_id>8705 AND t.kind_id=2 AND t.production_year=2006
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id=1 AND t.production_year=1913
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<235861 AND mc.company_id>13816
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id=101
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year>0
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<3642
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<6 AND ci.person_id<1443753 AND t.kind_id=7 AND t.production_year<1975
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>1676 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>1683102 AND t.kind_id=2 AND t.production_year=2004
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>2000
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=1 AND mk.keyword_id<1239 AND t.kind_id<3 AND t.production_year<2002
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<73 AND mc.company_type_id<2 AND mi.info_type_id>16 AND t.kind_id=2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<137402 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>8
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=8
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=42033
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND mc.company_id=38220 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>240 AND mc.company_type_id<2 AND t.kind_id>1 AND t.production_year<2010
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=6468
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>30180 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year<1993
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.kind_id=1 AND t.production_year<2009
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>11940 AND mi_idx.info_type_id>100 AND t.kind_id>4 AND t.production_year<2004
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.kind_id>3
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2925051
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2982758 AND ci.role_id=4
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=399606 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.kind_id=2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<1048
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=833367 AND t.kind_id=7 AND t.production_year=2010
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id<100 AND mk.keyword_id>56 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>3795 AND mk.keyword_id>5889
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id>2872001 AND ci.role_id<4 AND t.kind_id=6
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<2 AND t.production_year<2010
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2913042
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year>2013
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>91792 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<3164258 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2448882
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id=270441 AND ci.role_id<2 AND mc.company_id>2773
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>669985 AND ci.role_id>6
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>4 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>3268055 AND mi.info_type_id=16
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<83392 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2773316 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>4 AND mk.keyword_id<14026 AND t.kind_id=7 AND t.production_year>1987
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<101
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id>1 AND t.production_year=2005
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<3 AND t.production_year>2011
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>8190
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id>2909377 AND ci.role_id=3
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<527876 AND ci.role_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>17805
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id<7632 AND t.kind_id<7 AND t.production_year>2007
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=160
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=394 AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<910
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<11151 AND t.kind_id<7 AND t.production_year>2007
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>13
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=99 AND mk.keyword_id<1535
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=3219
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=48468
SELECT COUNT(*) FROM movie_info_idx mi_idx WHERE mi_idx.info_type_id<100
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>11219
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>7 AND t.kind_id<7 AND t.production_year<1978
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id=15581 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=192213
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<33512 AND mc.company_type_id=1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<5001
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1245121 AND ci.role_id=6 AND t.kind_id>1 AND t.production_year<1999
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=76111
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1075313 AND ci.role_id<10
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=7 AND t.production_year=2013
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id>1 AND t.production_year=2010
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id=3373016 AND ci.role_id<10 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>16604 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>73819
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>4 AND mi_idx.info_type_id>101
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<1987
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>617979
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id>7630
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id=892
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<234807 AND ci.role_id=3
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.kind_id=7 AND t.production_year=1958
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>3
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id=1510737 AND t.kind_id=7 AND t.production_year=1980
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<16 AND t.kind_id=7 AND t.production_year<2000
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND t.kind_id<3 AND t.production_year=2000
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>870
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<3 AND ci.person_id<524695
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2230364 AND ci.role_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<3576
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<17985 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>3415
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>3
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>228
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>4 AND t.kind_id<7 AND t.production_year<1929
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id<8558 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id=1094727 AND ci.role_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>317
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=707 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>819 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id>1 AND t.production_year>2012
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>2 AND mc.company_id>14048 AND t.kind_id<7
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year<2009
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id=11946 AND t.kind_id<7 AND t.production_year>1967
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<348 AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=4297
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=36664
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>1027
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1329504 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<150 AND mc.company_type_id>1 AND mi.info_type_id>18 AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>784
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=2
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id<2795459 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=141293
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<2115
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=3596159 AND t.kind_id=7 AND t.production_year<2014
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<20701
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<6 AND mk.keyword_id<1382 AND t.kind_id<7 AND t.production_year<1911
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=28445
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2720998
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<424
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<1562076 AND mc.company_id=521 AND t.kind_id>4
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<18
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>465
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>2827879
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>68
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id>13901 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<1135
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>3894
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1993262 AND ci.role_id>10
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<91361 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id>18060
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<6 AND t.kind_id>3 AND t.production_year=2010
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2888542 AND ci.role_id>4
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>2761076 AND ci.role_id>2 AND mi_idx.info_type_id<100 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1570491
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<14901 AND mk.keyword_id>58628 AND t.kind_id<7 AND t.production_year<1983
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>291101 AND mi.info_type_id=16
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2069715
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id<6 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<6
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<1061811 AND mc.company_id<73705 AND mc.company_type_id=1 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=1 AND ci.person_id<1862065 AND ci.role_id<3 AND t.kind_id<7 AND t.production_year=1993
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=3818 AND mc.company_type_id<2 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id<8
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<1244008 AND ci.role_id<2 AND t.kind_id>1 AND t.production_year<1996
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1672398 AND t.kind_id=1 AND t.production_year>2010
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<870
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=1578
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>1684
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=119 AND t.kind_id<4
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id=2570931 AND mi_idx.info_type_id<101 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.kind_id<4 AND t.production_year<2010
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<276753 AND ci.role_id=3 AND mk.keyword_id>2488 AND t.kind_id=7 AND t.production_year=2005
SELECT COUNT(*) FROM title t WHERE t.kind_id=1 AND t.production_year=2012
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<33850 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>7 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>15433 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year<2005
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<772 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>101 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=106
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<24382
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<4
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id<4918 AND t.kind_id<6
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<3416249 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=3
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<2
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<4
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id<4 AND t.production_year<2008
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=1 AND t.production_year=1999
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id>1 AND t.production_year<2013
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>12197
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>3
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>948 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=3
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8 AND mi_idx.info_type_id<100 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id=17533 AND mi.info_type_id<105
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<34346
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.kind_id<7 AND t.production_year>2005
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>72041 AND mc.company_type_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2966962
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2389043
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=7 AND t.production_year=2008
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=12807
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>2
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id>20344
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.kind_id=1 AND t.production_year=2011
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<20194 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND mi.info_type_id>3 AND t.kind_id>1 AND t.production_year<1984
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>15
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<1176043 AND ci.role_id=2
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id<13746 AND mc.company_type_id=1 AND t.kind_id<7
SELECT COUNT(*) FROM title t WHERE t.kind_id=1 AND t.production_year=2011
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>106 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<3897125
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year=2011
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<554 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2963171
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id>41459 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=71000
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=2
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<1760074 AND mi_idx.info_type_id<100
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year<2009
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=3263043 AND t.kind_id<7 AND t.production_year<2010
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<28038
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=19 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>166922
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>335
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1286822 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=35229
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<3256
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<5
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<121
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id>176960 AND ci.role_id>2
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>3154 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=4 AND t.production_year>0
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>1210 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=7 AND t.production_year=1987
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id<2156564
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<12888 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<2862
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=7 AND t.production_year<2005
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>11137 AND t.kind_id=7 AND t.production_year<1974
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<1135 AND t.kind_id<7 AND t.production_year>2005
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>864151 AND ci.role_id>1
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<2048390 AND ci.role_id=10 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=2593187 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<3
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=3891
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<14920
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>13015 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=1633 AND t.kind_id<7 AND t.production_year<2004
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<584675 AND t.kind_id<2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>317057
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id<11823 AND t.kind_id<7 AND t.production_year<1959
SELECT COUNT(*) FROM title t WHERE t.kind_id>4 AND t.production_year<1978
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year<2013
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year=2013
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=74330
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3 AND mk.keyword_id>1214
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<1322 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<2 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id=16 AND t.kind_id=7 AND t.production_year>2006
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3514064 AND ci.role_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<15 AND t.kind_id=7 AND t.production_year=1998
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id=905851
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2725737 AND ci.role_id>8
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>20622 AND t.kind_id=7 AND t.production_year>1987
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<619034 AND ci.role_id=2
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=1
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year=2013
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=1556
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id=11435 AND ci.person_id<2292389
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id=7 AND t.production_year=1957
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>23996 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id<101 AND mk.keyword_id<382 AND t.kind_id<4 AND t.production_year>2003
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2747449
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>1739
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id<2217 AND t.kind_id<2 AND t.production_year<1986
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1968421 AND ci.role_id>4 AND t.kind_id=7 AND t.production_year<2012
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND ci.person_id>3449725 AND t.kind_id=1 AND t.production_year>1957
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id<886118
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<672517
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND ci.person_id>912579 AND t.kind_id<7 AND t.production_year<2006
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>295
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<77431
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>145
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id<2399 AND t.kind_id>1 AND t.production_year=2000
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2171030 AND ci.role_id>2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<251919
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>531 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=2523
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3373670 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<394
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=18
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3480656
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<16
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<11344
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>1 AND t.production_year>1933
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<123 AND mk.keyword_id=1880 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>6 AND mc.company_type_id=2 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<7 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.kind_id=7 AND t.production_year<1981
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>1633 AND mc.company_type_id<2 AND t.kind_id=7 AND t.production_year=2010
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year=2012
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3162472
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>8
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2920204 AND ci.role_id>10
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>3636 AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>9445
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id<3346
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<596
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id<429 AND mc.company_type_id=2 AND t.kind_id=1 AND t.production_year<2012
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1748225 AND ci.role_id>3
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<192410 AND mi_idx.info_type_id<100 AND t.kind_id<7 AND t.production_year>2012
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1276636 AND ci.role_id<4
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>2369269 AND mi_idx.info_type_id=100
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>12651
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>126248 AND mk.keyword_id=55240
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=7 AND t.production_year<1989
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id>99
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND mk.keyword_id>1038 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id<49060 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>16
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3664683 AND t.kind_id>1 AND t.production_year>2012
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<7 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id<7 AND t.production_year>2006
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id>7076
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<15187 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=265
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>11 AND t.kind_id=1 AND t.production_year>1915
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id>687 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>101 AND mc.company_id>21353
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<914310
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id<98
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<3 AND t.production_year<2010
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>100
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<146108 AND mk.keyword_id=3434 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>12196 AND mk.keyword_id<2724
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<3 AND t.kind_id>1 AND t.production_year=1937
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>17 AND mk.keyword_id>7613
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>1258867 AND mi_idx.info_type_id=100 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id<30366 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<117
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id=15381 AND mk.keyword_id<323
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>19176
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<78326 AND mc.company_type_id<2 AND t.kind_id=1 AND t.production_year=1996
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id=1434 AND t.kind_id<7 AND t.production_year=1998
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=2047
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<8 AND mc.company_id>71609 AND t.kind_id=1 AND t.production_year=2011
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<329 AND mc.company_type_id=2 AND mi.info_type_id>8 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1704740 AND t.kind_id<7 AND t.production_year=1995
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>30465
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<7 AND t.production_year<2001
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=6 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>95 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1748205 AND ci.role_id>10
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id<39657 AND t.kind_id=1 AND t.production_year=1982
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id>7137
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<84760 AND mc.company_type_id>1 AND mk.keyword_id<12443 AND t.kind_id<3 AND t.production_year>2008
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id<7 AND t.production_year<1976
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3415552
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>1 AND t.kind_id=3
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1618613
SELECT COUNT(*) FROM title t WHERE t.kind_id=4
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<1043102 AND mi_idx.info_type_id=101
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>11153
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3177428 AND ci.role_id>2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<816
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=4
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id>1 AND t.production_year=1999
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2713276
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<172248
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=76
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8 AND t.kind_id<3
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1137697
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=1 AND t.production_year=2000
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=71236 AND mc.company_type_id>1 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id<101
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>798 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<33874 AND mi.info_type_id<8 AND t.kind_id=7
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year=1995
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id=3864758 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>5
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=12838
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id<7 AND t.production_year<2009
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>4 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1417352 AND ci.role_id>2 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id>1764101
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>686 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year=1995
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>85951 AND mc.company_type_id>1 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year>2012
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>7870
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=1758 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<27118 AND mc.company_type_id>1 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=27788
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>4610
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<31489 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=180218
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1784922
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>28787 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id<86012 AND mc.company_type_id>1 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<71479 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=754 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=21369
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.kind_id=1 AND t.production_year<1985
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<123706
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>2
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=7 AND t.production_year>2009
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>87163 AND t.kind_id>2 AND t.production_year>2008
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>1
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<17 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id<7 AND t.production_year<2003
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=2
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id=11243 AND mc.company_type_id=1 AND mk.keyword_id<412
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1250501
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=7 AND t.production_year<1996
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=17 AND t.kind_id>1 AND t.production_year=1991
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<3075 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>16 AND t.kind_id=7 AND t.production_year=1983
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=13812 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>5810
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<16 AND t.kind_id>4
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>1436 AND mk.keyword_id<3235
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<2174705 AND mi_idx.info_type_id=100 AND t.kind_id<4 AND t.production_year=2002
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=7 AND t.production_year=2010
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>19371
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=7 AND t.production_year=2013
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>2560706
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=19795
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<9766
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<2218539 AND ci.role_id=1 AND mi.info_type_id=7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.kind_id<7 AND t.production_year=1985
SELECT COUNT(*) FROM title t WHERE t.kind_id<3 AND t.production_year<1954
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2172640
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=461 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id>99 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id<2
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>8 AND t.kind_id=1 AND t.production_year>1964
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id=868 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id=180 AND mc.company_type_id<2 AND mi_idx.info_type_id=101
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id<3 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>580613 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>3542237 AND t.kind_id=4 AND t.production_year<2011
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>14731
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>3 AND t.kind_id=7 AND t.production_year=2011
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>7185
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>2712569
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.kind_id=2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<66914
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>594 AND mc.company_type_id=2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=2296891
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=1014398
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id>1 AND t.production_year>2003
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year<2012
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<7 AND t.production_year=1994
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id<7 AND t.production_year<2013
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id=689 AND ci.person_id<593777 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<1813082
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>70
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND mk.keyword_id=5098
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>7121
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<2586913 AND mi.info_type_id>18 AND t.kind_id>1 AND t.production_year>2006
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<5 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id>2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id=7 AND t.production_year>1922
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id<100 AND t.kind_id=2 AND t.production_year=2012
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>6 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id>16
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id>1 AND t.production_year>2010
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<3567550 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>7 AND t.kind_id<7 AND t.production_year>1999
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id<7 AND t.production_year=2006
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<19735
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id>335 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND mc.company_id=76662 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>13442 AND t.kind_id>1 AND t.production_year>2004
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.production_year=2006
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>3289625 AND ci.role_id>10 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>8 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<4022
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=7613
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>8 AND t.kind_id<7 AND t.production_year=2011
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1212357 AND ci.role_id=1
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id=3557553
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>1038
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1439361 AND t.kind_id=7 AND t.production_year>1991
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=3 AND t.production_year<1976
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<1047 AND t.kind_id=4
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id<7 AND t.kind_id=1 AND t.production_year>1951
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<3131616 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<13
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=1 AND t.production_year=2005
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<2561
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<5352 AND mc.company_type_id=2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3326049
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>74181
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id=99 AND t.kind_id>1 AND t.production_year>2012
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id<70 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=2784
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1233248 AND ci.role_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>895
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<1725
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<17206 AND mc.company_type_id<2 AND mk.keyword_id=348
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>4536
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>107 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2312135 AND ci.role_id=3
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1866830 AND ci.role_id<2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1380338
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1640784 AND t.kind_id=7 AND t.production_year<1994
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>99729 AND mi.info_type_id=6 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=11769
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year>2009
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=7 AND t.production_year<1972
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id<3 AND t.production_year>1981
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id<33242 AND t.kind_id<7 AND t.production_year<2011
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<2372710
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<7 AND t.kind_id<7 AND t.production_year<1974
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1215134
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id<3 AND t.kind_id>3
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=8637 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<4653
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<490 AND t.kind_id<7 AND t.production_year>2009
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<28437 AND mc.company_type_id=2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=527102
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<14587
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id>40599 AND t.kind_id=7 AND t.production_year<2008
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=48054
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id=66558 AND t.kind_id>1 AND t.production_year>0
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=2 AND mc.company_id>4625 AND t.kind_id<3
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<2953105 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<8605
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>2489 AND t.kind_id<2
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<9051 AND mi.info_type_id<16 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id=6 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND t.kind_id=2 AND t.production_year<2002
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=255 AND mc.company_type_id=2 AND t.kind_id<4
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<2 AND t.production_year>1962
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id<3 AND t.production_year<1965
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>16 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND mc.company_id<136093 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8 AND mi_idx.info_type_id<101
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<94769 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>3 AND mk.keyword_id>8821 AND t.kind_id=7 AND t.production_year<2010
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year=2000
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id=1764905 AND mi.info_type_id<2 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>8040
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=17849
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<10976
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1370359 AND ci.role_id<2 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>335
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>32913 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<5956 AND mc.company_type_id>1 AND t.kind_id<2
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id<6226 AND t.kind_id<7 AND t.production_year>1940
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id<2 AND t.production_year<2009
SELECT COUNT(*) FROM title t WHERE t.kind_id>4
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1026270 AND ci.role_id=1
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>98
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>6 AND mk.keyword_id<62841
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>64708
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id=4 AND t.kind_id>3
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>7 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=154139
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<694900 AND mc.company_id>74344 AND mc.company_type_id>1 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id>2725290
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>6210
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>71671 AND mc.company_type_id=2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id<7 AND t.production_year<2002
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year<2011
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id=2850979 AND t.kind_id<3
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>101
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<767198 AND ci.role_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id=7 AND t.production_year=2006
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<494
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=12144
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.kind_id>1 AND t.production_year>1995
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>75273
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=15
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id=762782 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>175411
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>3 AND ci.person_id<365791
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>206061 AND mi_idx.info_type_id>100
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id<7 AND t.production_year=1926
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<6 AND mc.company_type_id<2 AND t.kind_id<7 AND t.production_year=1999
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id>2350345 AND mc.company_id<190236 AND t.kind_id>4 AND t.production_year>2011
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<2652
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id>8 AND t.kind_id=3 AND t.production_year=2005
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id<98 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>1859538
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>101 AND mk.keyword_id>55944
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=639 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<7 AND mk.keyword_id<12218 AND t.kind_id<7 AND t.production_year=1978
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>18 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<10149
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND mc.company_id=76508 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id=127636
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND ci.person_id<2693073 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<3 AND t.kind_id=7 AND t.production_year<1993
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>1362114 AND mi.info_type_id<15 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=790666
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>98 AND mc.company_id>4743 AND t.kind_id>1 AND t.production_year<1973
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id=2980893
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<19 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<35157
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND mi.info_type_id<4 AND t.kind_id<7 AND t.production_year=2012
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=2238457 AND mk.keyword_id>4574
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>1993
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id=1588830 AND ci.role_id=1
SELECT COUNT(*) FROM title t WHERE t.kind_id=1 AND t.production_year<2006
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<13865
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id>1 AND t.production_year>1911
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1877378
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>7575 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id<2 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id>2080317 AND ci.role_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=41
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<861629 AND ci.role_id=2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=1 AND t.production_year=1925
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=100 AND t.kind_id<7 AND t.production_year>1986
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id<1741 AND t.kind_id<3 AND t.production_year>1917
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>100 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id>16
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id<12397 AND ci.person_id>1972789
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id=3 AND t.kind_id<7 AND t.production_year=1994
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<150787
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1190861
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<42387
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<359
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3555806 AND ci.role_id<10
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<7613
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>3 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id<3 AND mi_idx.info_type_id>99
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=3
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id<8960 AND mc.company_type_id=2 AND ci.person_id<2395521 AND t.kind_id=7 AND t.production_year<1956
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id=3451
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3674196
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<121379 AND ci.role_id<3 AND t.kind_id=1 AND t.production_year>0
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<105 AND t.kind_id<7 AND t.production_year<2008
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id<3 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=7 AND t.production_year>1930
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>4226
SELECT COUNT(*) FROM title t WHERE t.kind_id>2 AND t.production_year<1997
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>24387
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<943513 AND ci.role_id=2
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.kind_id=1 AND t.production_year=1982
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=5667
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id<666869 AND ci.role_id<8
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<843
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>444456 AND ci.role_id=1
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM title t WHERE t.kind_id=2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>21417 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id>807 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id>7777 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=6 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=2375
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>3 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id>1381
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>100225
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year<2008
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=12650
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=927
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year<1938
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>3303533 AND mi_idx.info_type_id=100 AND t.kind_id<7 AND t.production_year>2005
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<2743714 AND ci.role_id>2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<3181008
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>1426225 AND mi.info_type_id>105 AND t.kind_id>1 AND t.production_year>2001
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND mi.info_type_id=2 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<76143 AND mi_idx.info_type_id>99 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>114534 AND t.kind_id<7 AND t.production_year<1911
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id=7 AND t.production_year<2008
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<58647
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>19 AND mc.company_type_id=1 AND t.kind_id=1 AND t.production_year<1996
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>16 AND mk.keyword_id>733
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1439188 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>100257
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=15
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id=8 AND t.kind_id=6 AND t.production_year>2012
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>24335 AND t.kind_id=3 AND t.production_year<2011
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>98
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id<8
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>607489
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>5956 AND mc.company_type_id>1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>3580754 AND ci.role_id>1 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id=15 AND mi_idx.info_type_id=100 AND t.kind_id=7 AND t.production_year=1972
SELECT COUNT(*) FROM title t WHERE t.kind_id>4 AND t.production_year=1969
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=13815
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id=11764 AND t.kind_id=7 AND t.production_year>2006
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>5
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<853132
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id=1662 AND t.kind_id>2 AND t.production_year<2011
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2806269 AND ci.role_id=3 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year>1993
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year=1991
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<1402
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=6359 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<2523
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id=9174 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3629773
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>17 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id<7 AND t.production_year<1995
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>846
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=608953 AND ci.role_id>2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<20601
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2890656 AND ci.role_id>10
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND mk.keyword_id=1586 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id=1 AND mi_idx.info_type_id<101 AND t.kind_id>4 AND t.production_year<2004
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>14043
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<80617
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>323359
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>987845
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<494924
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>687 AND mc.company_type_id=2 AND mk.keyword_id>591 AND t.kind_id=7 AND t.production_year=2005
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id<72106
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>4147 AND t.kind_id<7 AND t.production_year<2003
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<3256
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1993842 AND ci.role_id=3
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<249
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=1 AND t.production_year<1994
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id<150
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year<2007
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2045593 AND ci.role_id=2
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>494 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<7 AND t.production_year=1958
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id<2185
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND ci.person_id>3213202
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>1049
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>5 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=71525
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=373
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2428128
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id=159
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>6411
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<867073
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<16 AND t.kind_id=1 AND t.production_year=2009
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year<1979
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1571459
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year<2008
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<28590 AND mk.keyword_id<16264
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<31431 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id<16 AND mk.keyword_id=27144 AND t.kind_id=7 AND t.production_year>1999
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>19275
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>74859
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id>16 AND t.kind_id>3
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<18
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=29205
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1723546 AND t.kind_id=7 AND t.production_year=2012
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3049826
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<28404
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<653921
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id>11203 AND mc.company_type_id=2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>234628 AND ci.role_id<3
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id=192061 AND mi.info_type_id=8
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>27927
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND t.kind_id>1 AND t.production_year>2009
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>41419
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year<2003
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<656930 AND ci.role_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<4 AND t.production_year>1994
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<1957
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<3909 AND mc.company_type_id=2 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3569558 AND ci.role_id>4
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id<2 AND t.kind_id>2
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=3664
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id<1615701
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<11 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=20778
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id>1 AND t.production_year=2012
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<104 AND ci.person_id>3904086 AND t.kind_id=1 AND t.production_year<2007
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>8 AND mc.company_id=54459
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<2665231 AND t.kind_id=7 AND t.production_year<2007
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1025921
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.kind_id>1 AND t.production_year<2012
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>2276
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year<1982
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2751370 AND ci.role_id>3
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<7278
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year=2000
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<105 AND t.kind_id>1 AND t.production_year<2004
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>99 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16 AND ci.person_id>3057916 AND t.kind_id<7 AND t.production_year<2010
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<17838
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.kind_id=1 AND t.production_year=2001
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1588038
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1465206
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>1764 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<76455
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=608486
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>2 AND t.kind_id>4 AND t.production_year<2001
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=1615 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=3 AND t.kind_id>3
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year>2005
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=98 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>31307
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND ci.person_id>1557098 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id>2276
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.kind_id=2 AND t.production_year=1937
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id>1 AND t.production_year=1996
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>37106 AND ci.role_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=1820
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id=4917 AND t.kind_id<7 AND t.production_year<1994
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3313429
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=883225
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2963556 AND ci.role_id<10
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>661640 AND ci.role_id<10
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=335
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND t.kind_id>1 AND t.production_year<2001
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id>1941
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.kind_id>4 AND t.production_year>2005
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<103336 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=4
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<6044 AND t.kind_id=1 AND t.production_year<2011
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>15
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=2841134 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<3139 AND t.kind_id<2
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year>2010
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>12677
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=93
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2234987
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>469698
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>164 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>100 AND mk.keyword_id<11210
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<5381 AND t.kind_id<3 AND t.production_year=1980
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3498848
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1419967
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.kind_id=2 AND t.production_year<1951
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1882291 AND ci.role_id>8
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id<2 AND t.production_year=1994
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<325798
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>27243
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=1 AND t.production_year>2000
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<2499516 AND mk.keyword_id=1608 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>155298 AND mi.info_type_id<4 AND t.kind_id<4
SELECT COUNT(*) FROM title t WHERE t.kind_id>2 AND t.production_year=1972
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=374580 AND ci.role_id>1
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id<101 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id>128078
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year=2003
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=536772 AND ci.role_id>3
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id>100 AND t.kind_id=3 AND t.production_year<2010
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>8 AND t.kind_id<7 AND t.production_year>2011
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND mc.company_id>868 AND t.kind_id>4 AND t.production_year>2009
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3481828 AND ci.role_id>3
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>7
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND ci.person_id<317778 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id=4 AND t.kind_id=4 AND t.production_year=2004
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<30517
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=169
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<15 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<15
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<9073
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id=3653222
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=4
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<48728
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>4520
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<80506 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND ci.person_id<61103 AND ci.role_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<992
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2139292 AND ci.role_id>10 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<10540
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=3223435 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.kind_id>3 AND t.production_year=2010
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3129866
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=11203 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id=1 AND t.production_year<2014
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>3139 AND t.kind_id=7 AND t.production_year<2010
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=68 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<160
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>594
SELECT COUNT(*) FROM title t WHERE t.kind_id>4 AND t.production_year>2008
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>7167
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id>2
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id>1 AND t.production_year>2006
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND mc.company_id>11203
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<3250949 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<1753
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id<545 AND t.kind_id>1 AND t.production_year<2008
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>16106 AND mc.company_type_id=2 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>19 AND mc.company_type_id>1 AND mk.keyword_id>7869
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id<2 AND t.production_year=1971
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<24091 AND mc.company_type_id>1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<2395777 AND ci.role_id=9 AND mk.keyword_id<9403 AND t.kind_id=7 AND t.production_year>2007
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3034708 AND ci.role_id=8
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=3567
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=186 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>19
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=78907
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<15005 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>15 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=7 AND t.production_year<1976
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<16 AND t.kind_id<7 AND t.production_year=1957
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3003783
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<76376 AND t.kind_id=7 AND t.production_year=1994
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>105608 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND t.kind_id<7 AND t.production_year=2002
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>3221
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>13780 AND mc.company_type_id=1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2704776
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=76379 AND t.kind_id<2 AND t.production_year>2009
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<3
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<2919093 AND t.kind_id>2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>15896
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<8992 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=4 AND t.kind_id<4 AND t.production_year>0
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>4625 AND mc.company_type_id=2 AND t.kind_id=1 AND t.production_year>1987
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=5138
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>114444 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=8
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=2891
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=20324
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<3 AND t.production_year=1935
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=460
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<1697
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>607 AND mc.company_type_id>1 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>6 AND t.kind_id<7 AND t.production_year<1996
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year=2011
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1793864 AND ci.role_id<4
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id<83216 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>1583
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id<2845196
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>295225 AND ci.role_id=1 AND mi.info_type_id=98
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<55711 AND ci.role_id=3
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=250764
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id<2762 AND t.kind_id=7 AND t.production_year>2004
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=8
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<2358
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<17460 AND mk.keyword_id<61147 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<3 AND mk.keyword_id>349 AND t.kind_id=6 AND t.production_year=2007
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<15713 AND t.kind_id<7 AND t.production_year=2010
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2952099
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<439
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1602067 AND ci.role_id<8
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<556500
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND ci.person_id=463645 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=18
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<16 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<436 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id=4
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id<7703 AND t.kind_id=1 AND t.production_year=1999
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2441057
SELECT COUNT(*) FROM title t WHERE t.kind_id>3
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>11414
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>104732 AND mk.keyword_id=15122
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id>3311 AND t.kind_id<3 AND t.production_year=2007
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<4 AND t.kind_id=3
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<44886 AND t.kind_id=7 AND t.production_year>1997
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=1304528 AND ci.role_id<3 AND mk.keyword_id<797
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id=4 AND t.kind_id=7 AND t.production_year<1988
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>1215469 AND mi.info_type_id>107
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id=2722174 AND mi_idx.info_type_id<100 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=10180
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>18 AND t.kind_id=1 AND t.production_year<2008
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<21366 AND mc.company_type_id=2 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id>2560908
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<42432 AND mc.company_type_id=2 AND t.kind_id=1 AND t.production_year=1943
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1765738
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<19 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<17
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1018677 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<1007539 AND ci.role_id=2 AND mc.company_id>89912 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id=4 AND t.production_year>2005
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<398 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<75454 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3953001 AND ci.role_id=10
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>14135 AND t.kind_id=7 AND t.production_year<2005
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=7 AND t.production_year>1989
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=94 AND ci.person_id>455054 AND ci.role_id>2 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<1592775 AND ci.role_id=10 AND t.kind_id<7 AND t.production_year>2006
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<15 AND t.kind_id<7 AND t.production_year=1964
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<11387 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>96038
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>864634 AND ci.role_id>1 AND t.kind_id=6 AND t.production_year=2004
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id<2830 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<2849
SELECT COUNT(*) FROM title t WHERE t.kind_id>3 AND t.production_year<2010
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>125126 AND ci.role_id>3
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id>3 AND t.production_year=1990
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<708415 AND t.kind_id<4
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=4 AND t.kind_id=7 AND t.production_year>2012
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=7 AND t.production_year<1962
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<6 AND t.kind_id<7 AND t.production_year<1973
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=502987 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id=2165 AND mk.keyword_id<1135
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<11205
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=105
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<2 AND mc.company_id>9749 AND mc.company_type_id>1 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=5
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=820845 AND t.kind_id<7 AND t.production_year<2006
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<2550
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=3658 AND t.kind_id>1 AND t.production_year<2006
SELECT COUNT(*) FROM title t WHERE t.kind_id=3
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<47
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>3665
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16 AND mc.company_id<22745 AND mc.company_type_id>1 AND t.kind_id<7 AND t.production_year=2012
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>2891029 AND mi_idx.info_type_id>99 AND t.kind_id=2
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>840
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=82098
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id>1 AND t.production_year=2000
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<36288
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>13751 AND mc.company_type_id=2
SELECT COUNT(*) FROM title t WHERE t.kind_id=1 AND t.production_year=2009
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<74675
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id<4
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id<6 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>101 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<702 AND mc.company_type_id<2 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2112537
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=15 AND mc.company_type_id=1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3682524
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id=3060547 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id=10705 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=7 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=88010
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=1 AND t.production_year=2009
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<140865
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<5608 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1299931 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id>1749
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>11
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<728492 AND ci.role_id=2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<2497
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<2797880 AND mc.company_id<46 AND t.kind_id>4 AND t.production_year<1975
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=130168
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id>1275
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id=11127 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id>696690 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>1406887 AND ci.role_id<3 AND mi.info_type_id<3
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>32670 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>2495474 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id>26023 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=64
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<89591
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>16692
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<11564 AND t.kind_id=7 AND t.production_year<1995
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>195208
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<7 AND t.production_year>2007
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3422685
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2918314
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<1369
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<390 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id=17460 AND mc.company_type_id=2 AND ci.person_id>302053
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<4 AND t.kind_id<7 AND t.production_year<2011
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1186472 AND ci.role_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=213686
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.kind_id=7 AND t.production_year=2010
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>6623
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=9 AND t.kind_id>2
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=4 AND t.kind_id=1 AND t.production_year>1988
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=7 AND t.production_year<1934
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>100 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<79056
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<2276 AND t.kind_id<7 AND t.production_year<1971
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=2 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=7 AND t.production_year>2011
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>16 AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=1578 AND t.kind_id<7 AND t.production_year<2009
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id<17 AND mk.keyword_id=7150 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=16 AND mk.keyword_id>117 AND t.kind_id<4 AND t.production_year=1977
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<265275 AND ci.role_id=10
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<11845
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>4 AND mi_idx.info_type_id=100 AND t.kind_id=7 AND t.production_year>1938
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1084144 AND ci.role_id<2
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND ci.person_id>2773626 AND ci.role_id<6 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<106 AND ci.person_id=120840
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=17150
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=7 AND t.production_year=2005
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id=3498223 AND ci.role_id=9 AND mc.company_id<1684
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=47752
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3183299 AND ci.role_id=1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<12306
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<97
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>4616 AND t.kind_id=4
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id<140734 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<18979
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>1049 AND mc.company_type_id>1 AND ci.person_id>2413515 AND t.kind_id=7 AND t.production_year=2005
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1058744 AND ci.role_id<4
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id=7851 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>100574
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=1 AND t.production_year<1971
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>2810854
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=11
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=10217
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3780628 AND ci.role_id>3
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id=2629
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>1513024 AND mk.keyword_id=16380
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<13255 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id=228 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<481412 AND mc.company_id>21092
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND ci.person_id<370733 AND t.kind_id>1 AND t.production_year>1952
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>1989
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>4 AND mc.company_id>12808
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id<1025 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1082849
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id<67642 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id=11652 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.kind_id<7 AND t.production_year<1985
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>160 AND t.kind_id<7 AND t.production_year>1966
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>2420798 AND mk.keyword_id<18362
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<2642630
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>16 AND t.kind_id<7 AND t.production_year>1975
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id>4590
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=12640
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id=115 AND mc.company_type_id=2 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<93725 AND mc.company_type_id=2 AND mi.info_type_id>15
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id>1 AND t.production_year=2011
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>13043
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id=1289062 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<124567 AND ci.role_id<8
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id>2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3103636 AND ci.role_id<10
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<16 AND t.kind_id=7 AND t.production_year>1993
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND mk.keyword_id=7168 AND t.kind_id<4
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1292281
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<390
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>7548 AND mc.company_type_id<2 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year>2006
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>11203 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=97938
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>349
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3004170 AND ci.role_id=10
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id<3 AND t.production_year<1993
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=13447
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=6300
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1895743
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<608399
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>872 AND t.kind_id=7 AND t.production_year>1980
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<3911461 AND mk.keyword_id>65842
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<7 AND ci.person_id<534219 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=2886441
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id<627302 AND ci.role_id>2 AND t.kind_id<3 AND t.production_year<1993
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>845
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year<2002
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id<7 AND t.production_year<1947
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1478505
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id>3186
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year<1911
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=6102
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=809 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>72374
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8 AND t.kind_id>1 AND t.production_year<2007
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id>5425 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<5 AND ci.person_id>1309033
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>521334
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<10805 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>3468 AND t.kind_id>4
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>398 AND t.kind_id<7 AND t.production_year>1981
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>1 AND t.production_year<2002
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=2965
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id>2585299
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND mi.info_type_id<5
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<7 AND t.kind_id<4
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<436 AND mc.company_type_id=1 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id>3257 AND t.kind_id=1 AND t.production_year=1915
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.kind_id=1 AND t.production_year=2010
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<2331433 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>76662 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id=2055381 AND ci.role_id>1 AND t.kind_id>1 AND t.production_year<2010
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=783
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>458799 AND t.kind_id=7 AND t.production_year<2005
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1440170 AND ci.role_id>1 AND t.kind_id<7 AND t.production_year=1951
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<395501 AND t.kind_id>1 AND t.production_year<1998
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<1451316 AND mk.keyword_id=137 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>16314
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id=5689 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id>139 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<2017 AND t.kind_id>2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<56
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1665294
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<160
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>580196 AND ci.role_id<2 AND t.kind_id<7 AND t.production_year>1971
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.kind_id=1
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year>1963
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<3219388
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1927460 AND ci.role_id>1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1686342 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<2680324 AND ci.role_id=1 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1820703
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id>2 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id<343541 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=15855
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND t.kind_id>1 AND t.production_year=2005
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>19 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id<18
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>11245 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id=3432150 AND ci.role_id<8 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id>4625 AND mc.company_type_id=2 AND t.kind_id>3 AND t.production_year<2007
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=99 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=728266 AND t.kind_id=7 AND t.production_year>1958
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1198222
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND t.kind_id=1 AND t.production_year>2010
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<132745 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1024640
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<86
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id=19 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id=15463
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<3073152 AND ci.role_id=1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<476247
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id>101
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id=1 AND t.production_year<1994
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id=1736 AND mc.company_type_id=1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2346383
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=1838656 AND mk.keyword_id<44816
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=40
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>5323 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id<2 AND t.kind_id=1 AND t.production_year<2004
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=1417
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>2456 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<4 AND t.kind_id=7 AND t.production_year=1947
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id=9 AND mi_idx.info_type_id=101 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3582159
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id<745057 AND ci.role_id<10 AND t.kind_id>1 AND t.production_year<1984
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND t.kind_id=7 AND t.production_year<2000
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=90 AND mc.company_id>38675 AND mc.company_type_id=2 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<21146 AND mc.company_type_id=1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1762080
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>8
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<2 AND mk.keyword_id<8821
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>5 AND ci.person_id<3019755 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3553115
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id>3 AND t.production_year>1988
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id=17
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.kind_id=7 AND t.production_year=2007
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<28339
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id=2103493
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id=19829 AND t.kind_id=3
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=3 AND t.production_year>2005
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id>3
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<5808
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year>2000
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id=2 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND ci.person_id<869626 AND t.kind_id=7 AND t.production_year>1976
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=17460 AND mc.company_type_id=2
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=3051300 AND mk.keyword_id<414 AND t.kind_id=7 AND t.production_year<1984
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id=2953 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1711416 AND ci.role_id<3
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=99 AND t.kind_id>4 AND t.production_year>2004
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>88664 AND mc.company_type_id>1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2480054
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>11278 AND mk.keyword_id>2199
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<11306 AND mc.company_type_id=2 AND t.kind_id>4 AND t.production_year<2012
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>2012
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=2192199
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND ci.person_id=1248691 AND ci.role_id<11 AND t.kind_id<7 AND t.production_year>1965
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=235
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=9
SELECT COUNT(*) FROM title t WHERE t.kind_id=1 AND t.production_year>2002
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id=2834159 AND ci.role_id>7 AND t.kind_id<2
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<15 AND ci.person_id=4058486
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>275771 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<12468 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND mi.info_type_id<8
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>674094 AND mi.info_type_id=8
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<160 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>2661
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<438
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=60773 AND ci.role_id=1
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<7 AND t.production_year=1975
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=2644891 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>93809
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=19
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id=65411 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=35396
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<196066 AND mc.company_type_id=1 AND mk.keyword_id=750 AND t.kind_id=3
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>10584 AND mc.company_type_id>1 AND mk.keyword_id>7467
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1499363
SELECT COUNT(*) FROM title t WHERE t.kind_id>4 AND t.production_year=2011
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<7489 AND t.kind_id=7
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year<2012
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<4
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1102140 AND ci.role_id<2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<589345
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=4447
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=1724 AND mc.company_type_id=1
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id=2474516 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<275
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=10675
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=10756
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<497933
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id=7838
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=11662
SELECT COUNT(*) FROM title t WHERE t.kind_id<2 AND t.production_year<2002
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<8294
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>95874 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=1 AND t.production_year=2003
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>25276
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id<7 AND t.production_year>2011
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=7 AND t.production_year=2005
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>16 AND mi_idx.info_type_id<101
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2930119 AND ci.role_id=8
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.kind_id=7 AND t.production_year>1980
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3477258 AND ci.role_id>2
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id>1 AND t.production_year<1956
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>6
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>1956
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=2505731 AND ci.role_id<5 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1920593 AND ci.role_id=2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=8
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND t.kind_id<7 AND t.production_year<2000
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<2650 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id<159018 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<31252
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3459890
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<107
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<40915 AND t.kind_id<7 AND t.production_year=1906
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<10822 AND mk.keyword_id=6039 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=26202
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>4
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<916309
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>81080 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=218710
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2976787 AND ci.role_id>10
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<3434
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=39656
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<355
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<625865 AND ci.role_id=10 AND mi.info_type_id<98 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2020457 AND ci.role_id<10
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<329
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id>4
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=4352
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>11146 AND ci.person_id<2780453 AND ci.role_id<2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<1416
SELECT COUNT(*) FROM title t WHERE t.kind_id<3 AND t.production_year=2003
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<457566 AND ci.role_id=3
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1734624
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<4568 AND t.kind_id=7 AND t.production_year=1981
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>517452 AND ci.role_id<2
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id<4 AND mk.keyword_id<1518
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2785360 AND ci.role_id>2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<3221
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=7 AND t.production_year=1982
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<19
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1166492
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=708
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1080347
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<94769
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1731853 AND ci.role_id>5
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>3 AND mk.keyword_id<3462 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id<7 AND t.production_year<1951
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id=85282 AND mk.keyword_id>398 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=13366
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>8782
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>2069
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<70477 AND ci.role_id=10 AND mk.keyword_id<2591
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<3988226 AND ci.role_id>3 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>94725
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<3171
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<442 AND t.kind_id=3
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<4
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<4025641 AND ci.role_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=2498
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year=1996
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<2424922 AND ci.role_id<3 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>18
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=6058
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id<49462 AND t.kind_id<7 AND t.production_year>1937
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>141779
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3982927
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>3256995 AND t.kind_id<7 AND t.production_year>2002
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>919936 AND ci.role_id>4
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<85976 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<6209 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1369039 AND ci.role_id>5
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND t.kind_id>1 AND t.production_year>2004
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>868
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>313 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>1749
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>7 AND t.kind_id=1 AND t.production_year>2011
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<6
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>253728
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=1733
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>241
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=3
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year<1992
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id<10081 AND t.kind_id=3
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=309 AND t.kind_id<7 AND t.production_year=1910
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>2007
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.kind_id<2 AND t.production_year=2005
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<15
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND mk.keyword_id=5159
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=16213
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>18826 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>3
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id>1708541
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<8529 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<699305
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id<16 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>1997374 AND t.kind_id>1 AND t.production_year<2009
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id<7 AND t.production_year=1973
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>2164689 AND t.kind_id<7 AND t.production_year=2006
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<16264
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=910 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>261
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101 AND mk.keyword_id=28728 AND t.kind_id<7 AND t.production_year>2008
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1791547
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id<3974 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id<7 AND mi_idx.info_type_id>101 AND t.kind_id=1 AND t.production_year=2009
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=42030
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>1998
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id=7428
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>2741232 AND t.kind_id=7
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>1979
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>3461061 AND mi_idx.info_type_id=100 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3094864
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=750
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>40229
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id>2488 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<18847
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id<4
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id<479 AND t.kind_id=7 AND t.production_year>2003
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>2 AND mc.company_id=329 AND mc.company_type_id<2 AND t.kind_id=1 AND t.production_year<2007
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=18430
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>2455
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year=1998
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>179685 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=73736 AND mc.company_type_id>1 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<11714
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<29549
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>100990
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>11257 AND mk.keyword_id>763 AND t.kind_id=1 AND t.production_year=2002
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<1873728 AND mi.info_type_id<98 AND t.kind_id=4
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=99 AND mk.keyword_id>10833 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=14423
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.kind_id>1 AND t.production_year>1974
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>19 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>71696 AND mi.info_type_id<16
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year>1973
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year>2002
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id>1046266 AND ci.role_id>2 AND t.kind_id=4
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id=2436
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=483
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<70 AND t.kind_id=7 AND t.production_year=2011
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=3660
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id<13642 AND t.kind_id=7 AND t.production_year<2006
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<299517
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=47
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<71443 AND t.kind_id>1
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year<1974
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=4 AND mk.keyword_id=12324 AND t.kind_id>1 AND t.production_year>0
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>9652 AND mk.keyword_id>11577 AND t.kind_id=2 AND t.production_year<2006
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=18628
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>5557
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>12978
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<27 AND mc.company_type_id>1 AND t.kind_id=7 AND t.production_year<2010
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>231024
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<15 AND mc.company_id<6 AND mc.company_type_id=1 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<4003018
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id=2965000 AND mi.info_type_id=7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>2 AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year=2004
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1406777
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=382 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=17042 AND mc.company_type_id>1 AND t.kind_id<7 AND t.production_year<2011
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<47
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>44736
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>5 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id<11 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3711753 AND t.kind_id=7 AND t.production_year>2010
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=14915
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id=7 AND t.production_year=1994
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1578377 AND ci.role_id<2
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<11952 AND t.kind_id>4 AND t.production_year>2003
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=8416 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>52480 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>689 AND mi_idx.info_type_id>99
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=7 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2589584
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id=5297
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=3032440 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>106121 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>2 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<3925 AND t.kind_id>1 AND t.production_year>2006
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.kind_id=7 AND t.production_year<2009
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.kind_id<7 AND t.production_year=2008
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>6 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3018253 AND ci.role_id<6 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id>1 AND t.production_year=1970
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>22664 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2663183
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=474374
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id<100
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1989012 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id=1 AND t.production_year=2000
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101 AND mk.keyword_id>31201
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<382
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>9 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<329 AND t.kind_id=1
SELECT COUNT(*) FROM title t WHERE t.kind_id=1 AND t.production_year=2007
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id=2587 AND mc.company_type_id=1 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<11145
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>14115 AND t.kind_id>3
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=8 AND mk.keyword_id=14587
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<5
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id=686 AND mc.company_type_id=2 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=16182
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<5521 AND mc.company_type_id=2 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>34481
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<293479 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>15
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id=1578
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<98959
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>1987
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=365 AND t.kind_id=7 AND t.production_year>2006
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id<7 AND t.production_year<2008
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id>611864 AND ci.role_id=2 AND t.kind_id<7 AND t.production_year=1968
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=374580
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3132179
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND t.kind_id=3
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<14890 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<16 AND t.kind_id<6
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>27 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>23403 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>16 AND t.kind_id=7 AND t.production_year>2005
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=7 AND t.production_year<1962
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id>4
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=3322950 AND ci.role_id>3
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=18964
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=360856
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1695675
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3168303 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id=9562 AND mc.company_type_id=1 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>4 AND t.kind_id=4 AND t.production_year<1982
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id=12193 AND mc.company_type_id<2 AND mi.info_type_id<16
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>70993 AND t.kind_id<7 AND t.production_year=2011
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2510039 AND ci.role_id=1 AND t.kind_id=4 AND t.production_year>2000
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>8
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<22626
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<43889 AND t.kind_id<4 AND t.production_year=2013
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<5830
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=1 AND t.production_year=2007
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id>3
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>7 AND mi_idx.info_type_id>99 AND t.kind_id>1 AND t.production_year>1993
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<65418 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<335
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<81
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<16 AND mc.company_id=89111
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>7 AND t.kind_id=7 AND t.production_year<1995
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id<8 AND mk.keyword_id>1002 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year=2001
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>28693
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1466446
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1041041 AND ci.role_id>9
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=8 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>911160 AND ci.role_id>10
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3091612
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id<4 AND t.production_year>2011
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1423112
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.kind_id<7 AND t.production_year<1994
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id=101 AND t.kind_id<7 AND t.production_year<2012
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id>2
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>16 AND t.kind_id=7 AND t.production_year>2004
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=141 AND t.kind_id>2 AND t.production_year<2011
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=69160
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id<117 AND t.kind_id<7 AND t.production_year=1999
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>183337 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id=11413 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>1680 AND ci.person_id<1626042
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1163418
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<117115
SELECT COUNT(*) FROM title t WHERE t.kind_id<2 AND t.production_year=1956
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>11172 AND t.kind_id=3 AND t.production_year<2012
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>574246
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<14405 AND mk.keyword_id=1860
SELECT COUNT(*) FROM title t WHERE t.kind_id>3 AND t.production_year>2009
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>2958280 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<3 AND t.kind_id<3
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<1649 AND mc.company_type_id>1
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<1597677
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id>1 AND t.production_year<1973
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<2644891
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<160601
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id<98584
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>7 AND t.kind_id=7 AND t.production_year=1997
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=1406 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=1378
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<13114 AND t.kind_id<7 AND t.production_year=2010
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>7158 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=12014
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=16519 AND t.kind_id=7 AND t.production_year>2011
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<820527
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=1427
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2602334
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=5 AND t.kind_id<2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1313876 AND ci.role_id>8
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=7 AND t.production_year>2003
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>6 AND mc.company_type_id=2 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<37748
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<11137 AND t.kind_id<7
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year=2009
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=76907 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND mc.company_id>71738 AND t.kind_id=7 AND t.production_year=2013
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year<2012
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<94957 AND mi_idx.info_type_id<101 AND t.kind_id<3
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1297340
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=7 AND t.production_year<2007
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=4 AND t.production_year>2009
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>39800
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>6893 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id=99 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<265720 AND ci.role_id>7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3127347
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND mi.info_type_id=16
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id=643 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3870997 AND ci.role_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id<1700 AND mc.company_type_id<2 AND t.kind_id<7 AND t.production_year>1992
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=948533 AND ci.role_id>3
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year=2007
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>28982 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>2050
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id<7 AND t.production_year=1999
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND mk.keyword_id<5890
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2781492 AND ci.role_id=4 AND t.kind_id<7 AND t.production_year>1967
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>58095
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=643 AND t.kind_id<7 AND t.production_year<2009
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>226295
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=7 AND t.production_year<2006
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<106 AND t.kind_id=1 AND t.production_year<2005
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<16 AND t.kind_id=7 AND t.production_year=2007
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=13
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=1884
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=48623
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id>9
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<797
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<400 AND t.kind_id=7 AND t.production_year<1952
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.kind_id<7 AND t.production_year=1991
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id=100 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id<7 AND t.production_year<1982
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id>18388
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>30652
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16 AND t.kind_id=3 AND t.production_year>2012
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>16 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=1475
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=7 AND t.production_year<2010
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>6
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id<517 AND mc.company_type_id<2 AND t.kind_id>1 AND t.production_year>1959
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<580 AND t.kind_id>2 AND t.production_year>2003
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>780809
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>3954 AND mk.keyword_id>359 AND t.kind_id>4
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=87986
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>18 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id>99 AND t.kind_id=3
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year<2011
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>101 AND mk.keyword_id>34 AND t.kind_id<7 AND t.production_year>1970
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<437569 AND ci.role_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<11961 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id>4
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<74469 AND mc.company_type_id>1 AND mi_idx.info_type_id<101
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>20330
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year=1998
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id>3810656
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year=2011
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=9005
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<657 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<900295 AND ci.role_id>2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>3093
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>1304810 AND ci.role_id=3
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=4 AND t.production_year>2010
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id<16 AND t.kind_id=4
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>272668 AND ci.role_id=1 AND mk.keyword_id=2488
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3200505
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=1 AND t.production_year>2010
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<106492 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=160 AND mc.company_type_id=2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1692347 AND ci.role_id=10
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id=101 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND t.kind_id<3 AND t.production_year<2002
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2838833 AND ci.role_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=774
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1880274 AND ci.role_id>6
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>292 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.kind_id>4 AND t.production_year=1987
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND mi.info_type_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>75614
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2470665 AND ci.role_id<2 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=3 AND t.production_year<1930
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>2010
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.kind_id=1 AND t.production_year<2013
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1696734 AND t.kind_id=7 AND t.production_year<1996
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id=2042 AND mk.keyword_id>2859 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>680 AND ci.person_id=1805013 AND ci.role_id<3 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=1 AND t.production_year<1997
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<7819
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1207383
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2981305 AND ci.role_id>2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=80196
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=16200 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=3 AND t.kind_id<7 AND t.production_year>1992
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=27457 AND ci.role_id>3
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=13543 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=2447
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id<667 AND t.kind_id=7 AND t.production_year<1978
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND mi.info_type_id=15 AND t.kind_id=2 AND t.production_year<2012
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>2176796 AND ci.role_id>4
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=4008335 AND ci.role_id=10
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=1 AND t.production_year<1980
SELECT COUNT(*) FROM title t WHERE t.kind_id<3 AND t.production_year=2011
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<98
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>1 AND t.production_year>1984
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id>2727 AND t.kind_id=1 AND t.production_year>2010
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=4778
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=93 AND t.kind_id=7 AND t.production_year<2012
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3336825
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>26598
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id=2
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<12618 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>27 AND mc.company_type_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1586900 AND ci.role_id<6
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<7 AND t.production_year<2011
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id>6 AND t.production_year<2005
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1869491 AND ci.role_id=10
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<21773
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id<5
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<984307 AND ci.role_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<9884
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>2614 AND t.kind_id<7 AND t.production_year<2007
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND mk.keyword_id<117
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<961924
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<12974
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=1567 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id<3471683 AND ci.role_id=2
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>103
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=1451 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>4654
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=86 AND t.kind_id=7 AND t.production_year>1988
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>27556
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<1128 AND mi.info_type_id>3
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<83263 AND mc.company_type_id=2 AND mi.info_type_id>7 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>83538 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>11649
SELECT COUNT(*) FROM title t WHERE t.kind_id=1 AND t.production_year=2004
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.kind_id<7 AND t.production_year>1987
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>467504 AND ci.role_id<3
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=1559
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<609886
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>105 AND mi_idx.info_type_id<100
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.production_year>2010
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>4021597 AND mk.keyword_id=365 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id=4929 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=13936
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id>105 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<27 AND mc.company_type_id>1
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>2005
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=24433
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<20408 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3730717
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND mc.company_id=53326
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND t.kind_id<7 AND t.production_year<1995
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<43311
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1674510 AND ci.role_id>6 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>101
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>1115861 AND ci.role_id<2 AND mi_idx.info_type_id<100 AND t.kind_id=7 AND t.production_year=2006
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=7 AND t.production_year=1995
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year>2007
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>1037
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>17 AND t.kind_id=4
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id>737561 AND ci.role_id<2 AND mc.company_id=42315 AND t.kind_id<3
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND ci.person_id<2687597 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>18 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=17
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<1969870 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=7 AND t.production_year>2005
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id<1410751
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>2887147
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=7 AND t.production_year>1988
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<3274127 AND mk.keyword_id<20601
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<3 AND t.production_year>2011
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id<4
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=1915
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id<11214 AND mc.company_type_id<2 AND t.kind_id=4
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>70864 AND ci.person_id>332449 AND ci.role_id>1 AND t.kind_id=7 AND t.production_year=1994
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>336150
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<197740 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<52922
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=3 AND t.kind_id=2 AND t.production_year=2010
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>78142
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2715346
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>80014
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2009295 AND ci.role_id=4 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<8 AND t.kind_id<7 AND t.production_year<1953
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>67
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<4030
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3358061
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<35517 AND mi_idx.info_type_id=101
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<8 AND t.kind_id<3 AND t.production_year>1988
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2978458
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=423 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<266
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND t.kind_id<7 AND t.production_year>1978
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1418033 AND ci.role_id<4
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id=2 AND t.production_year>2006
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id=4515 AND mi_idx.info_type_id=99
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id=2112280 AND mk.keyword_id>3991 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=4 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<12242 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=425
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.kind_id=7 AND t.production_year>2013
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>15 AND mc.company_id>83622
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>244831 AND ci.role_id>9
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>572497 AND ci.role_id>2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>78115
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.kind_id<4
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id=359 AND t.kind_id=7 AND t.production_year=1961
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id=3
SELECT COUNT(*) FROM title t WHERE t.kind_id>2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=382
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<32316
SELECT COUNT(*) FROM title t WHERE t.kind_id=1 AND t.production_year<2012
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<2 AND t.production_year<2014
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id>28647 AND t.kind_id=6
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<20064
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<846 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id>1 AND t.production_year>2012
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id=1684979 AND ci.role_id>1 AND t.kind_id>3
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<73807 AND t.kind_id<4
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id>2 AND t.production_year<2007
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>43739 AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>1138
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<165119 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id>664619 AND ci.role_id=5 AND mc.company_id<116605 AND mc.company_type_id=2 AND t.kind_id>1 AND t.production_year<2010
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND t.kind_id=2 AND t.production_year>2012
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=102808
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND t.kind_id=2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<2488
SELECT COUNT(*) FROM title t WHERE t.kind_id=3 AND t.production_year=1961
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<4978
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<2 AND t.kind_id<7 AND t.production_year<2007
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>5376 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id=868 AND t.kind_id=3 AND t.production_year>1913
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<71767 AND mk.keyword_id=309 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>820935 AND ci.role_id=4 AND t.kind_id=7 AND t.production_year=2010
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>180163 AND ci.role_id=1 AND t.kind_id<2 AND t.production_year>2008
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>16 AND t.kind_id=7 AND t.production_year>1933
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2732608
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>94
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year<2010
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<564817 AND ci.role_id=10 AND mi.info_type_id<13 AND t.kind_id>1 AND t.production_year>2009
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id=7
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<1424446
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=1 AND t.production_year<2000
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<3
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<25206 AND t.kind_id<7 AND t.production_year>1985
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<2 AND mc.company_id=22064
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<178 AND mc.company_type_id=2 AND mk.keyword_id<305 AND t.kind_id=7 AND t.production_year=1997
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=100 AND t.kind_id<7 AND t.production_year>2010
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<1729068 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id>2197 AND t.kind_id<3
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id=500774
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id=517097 AND ci.role_id<10 AND mi.info_type_id<2
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=106 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=14444
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2771865 AND ci.role_id=1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<321046
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=9667 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<170233 AND mc.company_type_id=2 AND t.kind_id=7 AND t.production_year=1977
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>1962
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND mc.company_id=1784
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id>6226 AND t.kind_id>1 AND t.production_year>1976
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<49739 AND mc.company_type_id>1 AND mk.keyword_id=63248 AND t.kind_id=7 AND t.production_year<2013
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<2718828 AND ci.role_id<2 AND t.kind_id>3 AND t.production_year<2002
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>2237915 AND ci.role_id=9 AND mk.keyword_id<42252 AND t.kind_id<7 AND t.production_year=2012
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id=112076
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<83703 AND t.kind_id>2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1649086
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1066461 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>18084 AND mc.company_type_id<2
SELECT COUNT(*) FROM title t WHERE t.kind_id<2 AND t.production_year<1982
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id>2
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id>1 AND t.production_year<1993
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=4 AND t.production_year>1983
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=8654
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=2412019 AND t.kind_id>1 AND t.production_year=2010
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=21304
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=18 AND t.kind_id=7 AND t.production_year<2006
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>3401
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<3
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>8 AND t.kind_id<7 AND t.production_year=1920
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1874310
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<2940221 AND ci.role_id=2 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=481682 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=99 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.kind_id=1 AND t.production_year=2013
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id>98
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>71605 AND t.kind_id=1 AND t.production_year>1969
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year=1974
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id=11414 AND t.kind_id=7 AND t.production_year>1942
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<265 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<2102
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>405
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<31 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=3
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<12108
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=18 AND ci.person_id>1569389
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<1220
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id>30631 AND t.kind_id<7 AND t.production_year=1939
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.kind_id=1 AND t.production_year<2006
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>101 AND t.kind_id<7 AND t.production_year<2007
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id<3537 AND t.kind_id=1 AND t.production_year=1997
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>1756145 AND ci.role_id>2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1496177 AND ci.role_id>2 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3042082
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=15 AND t.kind_id>4
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.kind_id<7 AND t.production_year<1910
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>348
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id<15351 AND t.kind_id=4 AND t.production_year<1999
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=38427 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id>17367
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<195053
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id=214808 AND ci.role_id=1 AND mi_idx.info_type_id>100 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<16264 AND t.kind_id=1 AND t.production_year=2006
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>1991
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>419 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1202995
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=4837
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>101 AND mi.info_type_id<16 AND t.kind_id<7 AND t.production_year>1997
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1052691 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>5403
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id<1060 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>15 AND ci.person_id>3023749 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id=540
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=192819
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1470428
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<1202475 AND ci.role_id>1 AND t.kind_id<7 AND t.production_year>1967
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>107
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id>4 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<76
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=3369 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>4047725
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id>1 AND t.production_year<2006
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id=1 AND t.production_year>1999
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<1310559 AND ci.role_id>4
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<784
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>122314
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=5300
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<251930 AND mi.info_type_id=3 AND t.kind_id>1 AND t.production_year>1985
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2048976
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<7639
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=2067
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<16 AND t.kind_id<7 AND t.production_year>1937
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<30822 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id>1 AND t.production_year<2000
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>1721662
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<1709 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=4 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1562869 AND ci.role_id=1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id<7 AND t.production_year>2005
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.kind_id<2 AND t.production_year=2008
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year<2007
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.kind_id<7 AND t.production_year<2011
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<1714 AND mc.company_type_id=2 AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>1941
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3 AND mc.company_id<34 AND t.kind_id<7
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year<2008
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id=439 AND mc.company_type_id=2 AND mi_idx.info_type_id=101 AND t.kind_id<7 AND t.production_year>2012
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1829036
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id<2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2164409 AND ci.role_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year>1915
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id<77390 AND mc.company_type_id<2 AND t.kind_id=7 AND t.production_year=2001
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>43011
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2405334 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<3256874 AND ci.role_id=4
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=13
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id>2021323 AND ci.role_id<10
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<48940
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<47428 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id>3153795
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<180
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<789 AND mk.keyword_id<7829 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>8245
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<38756
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8 AND ci.person_id<2045135 AND ci.role_id=6
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1433572
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3770034
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<3157016 AND t.kind_id>1 AND t.production_year<2008
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=897 AND mc.company_type_id=1 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=652909
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND mc.company_id=91395
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id>90 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id>1 AND t.production_year>1995
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<2761
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<2 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id=3463 AND t.kind_id<7 AND t.production_year>2011
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id=1510062 AND mi.info_type_id=8
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id<685 AND mc.company_type_id<2 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>2036
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<75807
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>8426
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=64083
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id=3067
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id<71059
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3051454 AND ci.role_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<8 AND t.kind_id>1 AND t.production_year<2000
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=29318
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1859840 AND ci.role_id<11
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>447439
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2802568
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1171903 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=5
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>7634
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=13681
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<3810217 AND mc.company_id<56018 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=101
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>3 AND t.kind_id=7 AND t.production_year>2004
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>17 AND t.kind_id=3
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year<1974
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>4 AND t.production_year=2006
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<8604
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3728619 AND ci.role_id>8
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<17 AND t.kind_id<7 AND t.production_year=2003
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=14568
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.kind_id=4
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<15789 AND t.kind_id>2 AND t.production_year>1972
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<2021 AND mc.company_type_id=1 AND mk.keyword_id>2770
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>71884 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<7214
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1306294
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2394545 AND ci.role_id<2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>101
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<129522
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>101 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id<7323 AND mc.company_type_id>1 AND t.kind_id=7
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year=1985
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id<101 AND mk.keyword_id<5808 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year=1964
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=3448
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=100
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>242 AND t.kind_id>3 AND t.production_year>1996
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id>67092
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16 AND mc.company_id=11149 AND mc.company_type_id>1 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>2529
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=728
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<31468
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<2 AND t.production_year<2013
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id=16 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND mc.company_id>34374 AND mc.company_type_id>1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>204021
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<46454
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=36755 AND mc.company_type_id<2 AND t.kind_id=1
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year=2011
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=88253
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>7517 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<68212 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1563038 AND t.kind_id>1 AND t.production_year>2006
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<1435
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<137
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=1 AND t.production_year<2000
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1145735 AND ci.role_id<3
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id>17 AND t.kind_id>1 AND t.production_year<2010
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<105
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>2174623
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<2689907 AND mi.info_type_id<16
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>1996
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<98
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=102051
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3055800
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id>162136 AND ci.role_id>3 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id=101568
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.kind_id=1 AND t.production_year=2009
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id=1 AND t.production_year<2006
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<181 AND mk.keyword_id>5415 AND t.kind_id<7 AND t.production_year>2006
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>1495180
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=10112 AND t.kind_id<3
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<631755
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id>3540 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<9749 AND mc.company_type_id=2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>812251
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>16 AND t.kind_id<6 AND t.production_year<1953
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<195926 AND ci.role_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>88462
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=7 AND t.production_year>2012
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>294597 AND ci.role_id>6
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=13669
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id=10078
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>100 AND mk.keyword_id=500
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id<12135 AND mc.company_type_id>1 AND ci.person_id>1238278 AND ci.role_id=2 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16 AND ci.person_id<3373083 AND ci.role_id<4 AND t.kind_id=4
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id<3314
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id<7 AND t.production_year>1989
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<35
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id>2867 AND t.kind_id<7 AND t.production_year>2008
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id<129180
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=1 AND t.production_year=2004
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND mk.keyword_id=73187
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<2721047 AND ci.role_id=2 AND mc.company_id=5690 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<1812353 AND ci.role_id=1 AND mc.company_id>44814 AND mc.company_type_id>1 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<46 AND t.kind_id=7 AND t.production_year<1970
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>3337
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<117 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=7238 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id<16 AND t.kind_id<7 AND t.production_year<2004
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=65213 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year=2010
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>4547
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year=2005
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=20978
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<225398 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>16 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2053814 AND ci.role_id=2 AND t.kind_id=7 AND t.production_year=1994
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>99 AND mk.keyword_id<569 AND t.kind_id=7 AND t.production_year=2004
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=98 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<2571423
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>66207
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1668029 AND ci.role_id>1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>58856 AND t.kind_id=1 AND t.production_year>2005
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=49
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=7 AND t.production_year>1982
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>17 AND mk.keyword_id>212
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<4469
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3 AND ci.person_id<2120221 AND t.kind_id<7 AND t.production_year=2013
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.kind_id<7 AND t.production_year>2003
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<16264 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=2931
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.kind_id<4 AND t.production_year=1917
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id>11714 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<153144 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=459
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id<2 AND t.production_year=2001
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2865703 AND ci.role_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=1987
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<2487097 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1582623
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=411 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3836477
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id<13395
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id>6 AND t.kind_id<4 AND t.production_year<2009
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=1 AND t.production_year=2008
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3119982 AND ci.role_id>10
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<92537 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id<7 AND t.production_year=2005
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2116659
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=88243
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.kind_id=1 AND t.production_year<2000
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=100 AND t.kind_id=2 AND t.production_year<2012
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id<4976 AND t.kind_id=1 AND t.production_year>2010
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>2177738 AND ci.role_id=8 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>1479 AND mc.company_type_id=2 AND mk.keyword_id<12106 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<733 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<29028
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>54070
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>8651
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.kind_id>3 AND t.production_year=1969
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.kind_id=7 AND t.production_year=1997
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year>2003
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.kind_id=1 AND t.production_year>1948
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=18785 AND t.kind_id=1 AND t.production_year>2010
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id>23375 AND t.kind_id>1 AND t.production_year=2008
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=769 AND t.kind_id=7 AND t.production_year<1962
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=7 AND t.production_year=2008
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id>2762
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id<16432
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id=359 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1661438 AND ci.role_id=9
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=137
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id=12194
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2513654
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<13704
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=2 AND t.production_year<2011
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id=159 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=211
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id<3171 AND mc.company_type_id>1 AND t.kind_id<7 AND t.production_year=2012
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=83203
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1214932 AND ci.role_id=1 AND t.kind_id=7 AND t.production_year<2003
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3497830 AND t.kind_id<7 AND t.production_year=2011
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>1639 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=3450729 AND mk.keyword_id<2069
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id=30251
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>1226
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id>1 AND t.production_year>1978
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id<16 AND mi_idx.info_type_id>100 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3296372 AND ci.role_id>3
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=71440
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<74027 AND t.kind_id=7
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year<2005
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>150
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id<7 AND t.production_year<1977
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=144151
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id<1374417
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND mi.info_type_id=6 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<104926 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id=25896 AND mi_idx.info_type_id=99
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1320605 AND t.kind_id=2
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id=4 AND t.kind_id=7 AND t.production_year<2013
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>11147
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<15328 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND mk.keyword_id=3673
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<3247
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND mc.company_id=71831 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<20335
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.kind_id=1 AND t.production_year>2002
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id<16
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=858976
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=1439
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id>100
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<94265
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<37674
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id>4 AND t.production_year<1942
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id>1349047 AND ci.role_id>3 AND mc.company_id>11852 AND t.kind_id=1 AND t.production_year=2004
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>5569
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id<11647
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id<63072 AND t.kind_id=7 AND t.production_year>2009
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<1352842 AND ci.role_id>1 AND mk.keyword_id<7991 AND t.kind_id=7 AND t.production_year<2009
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<3
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1740342
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id<18688 AND t.kind_id=1 AND t.production_year>2011
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year<1996
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id<8 AND mi_idx.info_type_id>99 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=1083
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>108486 AND t.kind_id=1 AND t.production_year=1998
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=73526 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>5297
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>18
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<106008
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id=30415 AND mc.company_type_id<2 AND ci.person_id<2257348 AND ci.role_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>1102 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id>4 AND t.production_year>2005
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2202386 AND ci.role_id<9
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>74142 AND t.kind_id=7 AND t.production_year>1913
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year=2003
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<81241 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<50162 AND mc.company_type_id=2 AND mk.keyword_id=13572 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<1258
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id<7 AND t.production_year<2008
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1564611 AND ci.role_id>1 AND t.kind_id=7 AND t.production_year>2005
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>4 AND mi_idx.info_type_id=99
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id>4
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>329881
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1303945 AND ci.role_id<4
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>155772 AND mi.info_type_id=7 AND t.kind_id=7 AND t.production_year<2011
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<19 AND t.kind_id=1 AND t.production_year>2010
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id=558830 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id<100 AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>7984
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year=2008
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1035086
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>23826 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=14069
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id=7 AND t.production_year<2005
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=16264
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year<2010
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3510085
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>16443
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id=11
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>7845 AND t.kind_id<7 AND t.production_year<1964
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=538599
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id<7 AND t.production_year>2003
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<65987
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id=4038 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id<374000 AND ci.role_id>10 AND t.kind_id=7 AND t.production_year=2007
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id=13614 AND mc.company_type_id>1 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=7 AND t.production_year=2001
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1839038
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>90 AND t.kind_id=7 AND t.production_year<2011
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=4 AND t.production_year<2012
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2093131 AND ci.role_id=3 AND t.kind_id<4 AND t.production_year=2004
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<1863850 AND ci.role_id=2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>11137
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<3587681 AND mi.info_type_id=106 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND mi.info_type_id>7 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=4
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>11369
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<2175505 AND ci.role_id<10 AND mi.info_type_id=13
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=2409
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3905105
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=3959683
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id>2629841
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<580068
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<7373
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3180594 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=6 AND t.kind_id<4 AND t.production_year<1974
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2033019
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id>13084
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<24015
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=11139 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=4029374
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>2198
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<65033
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>55865
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=7 AND mc.company_type_id=1 AND t.kind_id>1 AND t.production_year>1991
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year>1989
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=1 AND t.kind_id=7 AND t.production_year>2012
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>6967
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>865
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id>3 AND t.production_year=2007
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>59991
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1606595 AND ci.role_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>160
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id=7 AND t.production_year<1991
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<7 AND t.production_year=2009
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2278392 AND t.kind_id>1 AND t.production_year=2011
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>107 AND mc.company_id>15100 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>8 AND mc.company_id=23402 AND t.kind_id<7 AND t.production_year=1984
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=3375
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<851995 AND ci.role_id=3
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year>1995
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1174845 AND ci.role_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=127
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id=3348402 AND ci.role_id>4 AND t.kind_id<7 AND t.production_year>0
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3017011
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>4 AND t.production_year=1997
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<1159 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<285986 AND ci.role_id<10 AND t.kind_id=1 AND t.production_year<2010
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=7 AND t.production_year>1930
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<503836
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=100 AND t.kind_id=2 AND t.production_year<1997
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=74905
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id<7 AND t.production_year>1975
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>1436565 AND ci.role_id=1 AND t.kind_id<7 AND t.production_year>1963
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>8761 AND mc.company_type_id=2
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id>18 AND t.kind_id=3
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=4350
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<135470 AND t.kind_id=7 AND t.production_year<2004
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>47111
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>1522002 AND ci.role_id>1 AND mi_idx.info_type_id=99 AND t.kind_id=4 AND t.production_year<2011
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id<21386 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<9897
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<3222769 AND ci.role_id=10 AND t.kind_id<7 AND t.production_year>1971
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3690900 AND ci.role_id>1 AND t.kind_id>1 AND t.production_year=1964
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>2234401 AND ci.role_id>1
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>4 AND ci.person_id<1073251 AND t.kind_id=1 AND t.production_year=1981
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<403
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=19621
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=7 AND ci.person_id>1289037 AND ci.role_id=9
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>11205 AND mc.company_type_id=1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2589937
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>1820 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=12509 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=6
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3412324
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1207774 AND ci.role_id<10
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=18 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<867078 AND ci.role_id=1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=7 AND t.production_year>1986
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=10374
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3674765
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year<1998
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>131145
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=3
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=48612
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id>106 AND t.kind_id=1 AND t.production_year=2008
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id>128645 AND ci.role_id<4
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3 AND mk.keyword_id>2860 AND t.kind_id=7 AND t.production_year<1953
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=8434 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=2 AND mc.company_id>3383 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<98 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>1009916 AND ci.role_id>1 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND t.kind_id<7 AND t.production_year=2011
SELECT COUNT(*) FROM title t WHERE t.kind_id<4 AND t.production_year>1999
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16 AND mc.company_id<1378 AND t.kind_id<7 AND t.production_year>2001
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>245505 AND ci.role_id<9
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1849223
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=11373
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=660139
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<258
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id<101
SELECT COUNT(*) FROM title t WHERE t.kind_id=4 AND t.production_year>2006
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id>225 AND mc.company_type_id<2 AND t.kind_id>1 AND t.production_year<1999
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id<4 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<11313
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2024249
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=11379
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id<101 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>8 AND mk.keyword_id<72217
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<26107 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<11264
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3 AND ci.person_id<1099815 AND ci.role_id>1 AND t.kind_id<7 AND t.production_year<1917
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=2
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id=99
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year=1990
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101 AND mk.keyword_id=10049
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=1512
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<76
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<23847 AND mc.company_type_id=2 AND t.kind_id=7 AND t.production_year<2012
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=71259
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND mk.keyword_id>245
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1026524 AND ci.role_id<10 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>5148
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1771810
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=22057
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year>1975
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<263687
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<10018
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<45913 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>519
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id<105 AND mi_idx.info_type_id=101 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=1711
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>63 AND mk.keyword_id=16532
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>50752 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>38894 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<63 AND t.kind_id=1 AND t.production_year<2011
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<394691 AND ci.role_id=4
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>2006
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>304689
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1710632 AND ci.role_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<34307
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=107
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>117966 AND t.kind_id=3
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<1390850 AND ci.role_id<8
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3006980
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id<9545 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>82155
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id>924444 AND ci.role_id=1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3092106
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>107 AND t.kind_id<7 AND t.production_year<1980
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=857
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.kind_id<4
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<15 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>1046
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=3
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=13580
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id=21730 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=2193
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=9826 AND ci.role_id<6
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>2069
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<73111 AND mc.company_type_id=2 AND mk.keyword_id>1382 AND t.kind_id<6
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<569918 AND ci.role_id<3 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=7 AND t.production_year=1994
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id>4328 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=1 AND t.production_year<2006
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>25676
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<16 AND t.kind_id<4
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=4 AND t.kind_id=4
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id=1 AND t.production_year=2010
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<2
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year<2009
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<6967 AND mc.company_type_id<2 AND t.kind_id=7 AND t.production_year>2004
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year>2008
SELECT COUNT(*) FROM title t WHERE t.kind_id<2
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<86223
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=13565
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<1114
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND ci.person_id<559241 AND t.kind_id=1 AND t.production_year<2007
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=7 AND t.production_year<2004
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>8 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=107
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<901406 AND ci.role_id>3 AND mk.keyword_id>11694 AND t.kind_id=7 AND t.production_year<2004
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2204210
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND ci.person_id>179647 AND t.kind_id=7 AND t.production_year<2004
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id>308951 AND ci.role_id=4
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.kind_id>3 AND t.production_year>2008
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>106 AND t.kind_id=1 AND t.production_year>1969
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<3249279 AND ci.role_id>1 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=7 AND t.production_year>2013
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<1732
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=71095
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=291190
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=12694
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year>2001
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>382
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND mc.company_id=13642
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id<6 AND t.kind_id<7 AND t.production_year<2007
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id>8654
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<2503
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>8 AND mi_idx.info_type_id<101 AND t.kind_id=3 AND t.production_year=2005
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=11 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<4 AND t.kind_id<3
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=6 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id>4 AND t.production_year=2010
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id<7 AND t.production_year<1972
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<17690 AND mk.keyword_id<84093 AND t.kind_id>1 AND t.production_year=2012
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2201612
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<607
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id=7 AND t.production_year<2001
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16 AND mi_idx.info_type_id>99 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<13618
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id<108395 AND ci.role_id>4 AND t.kind_id<7
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>2011
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id=1457719
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id<7771 AND t.kind_id=1 AND t.production_year<1980
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3 AND t.kind_id<3
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3813503
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=2 AND t.production_year>2008
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>1094514 AND mi.info_type_id=4 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=26256
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year<2008
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<15 AND mc.company_id<84094 AND t.kind_id=7 AND t.production_year>2006
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>31716
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=121
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<31525
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>10847 AND mc.company_type_id>1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3296482
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=8424 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<12579 AND mc.company_type_id=2 AND mk.keyword_id<826 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=7 AND t.production_year>1978
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>816
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<27572 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<12740 AND mc.company_type_id<2 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>3294
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>2032623 AND mi.info_type_id=16
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.kind_id>4
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id<3 AND t.kind_id<6 AND t.production_year=1968
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=6
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id>1 AND t.production_year<1974
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2810005
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>11554
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<2245189 AND mc.company_id=11137 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>342154 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=27233
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=423052 AND mk.keyword_id=117 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>15108 AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=115
SELECT COUNT(*) FROM title t WHERE t.kind_id=1 AND t.production_year=1961
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year>1985
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<3157001
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=14400 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<513 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>100070
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1468398
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id<21183
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id>1 AND t.production_year=2011
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<531064 AND t.kind_id<7 AND t.production_year<1954
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>98
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND ci.person_id<926815 AND ci.role_id<8
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<1451 AND mc.company_type_id>1 AND t.kind_id<7 AND t.production_year=1999
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3048011 AND t.kind_id<7 AND t.production_year=2011
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=11853 AND t.kind_id<7 AND t.production_year<2009
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id>100
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id=884722
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<60498 AND mc.company_type_id>1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id=12080 AND t.kind_id<2
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=149698
SELECT COUNT(*) FROM title t WHERE t.kind_id>6 AND t.production_year<2010
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<6429 AND mk.keyword_id>5279
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>1686960 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>7
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id>5
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=7 AND t.production_year<1911
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<1569 AND mc.company_type_id<2 AND mk.keyword_id>14360 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>19701 AND ci.person_id>827039
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>6
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=1645856 AND t.kind_id=1 AND t.production_year>1987
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id=4019402 AND mi_idx.info_type_id>99
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>16 AND ci.person_id<2532951 AND ci.role_id>2 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>6667 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<837
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year<1985
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<41435
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=1 AND t.production_year=2010
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=717135 AND t.kind_id<4 AND t.production_year=2007
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>2761
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id<1483453 AND ci.role_id=3
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1328922
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1196927 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id<2 AND t.production_year<1991
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=7464
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>16 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>2 AND t.kind_id<7 AND t.production_year>2009
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=3401
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1745553 AND ci.role_id>3 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>567616 AND ci.role_id>10 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<6857
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=12648
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2443579 AND ci.role_id=10
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=3 AND t.kind_id=2
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>177933 AND t.kind_id>3
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16 AND mc.company_id<3065 AND t.kind_id=7 AND t.production_year<2006
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<10114 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=11600
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>166 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>19 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id<18 AND t.kind_id=2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3199881
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<13886
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<2455206
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id>2 AND t.production_year=2008
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<508
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<6 AND mc.company_id<189 AND mc.company_type_id=1 AND t.kind_id<7 AND t.production_year<2010
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=71121 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<16708 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>4016
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<7365 AND t.kind_id=4
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>7663
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<3262
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>830
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>101 AND mc.company_id>18877
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=20482
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id=2840
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=6783
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>234245 AND ci.role_id=1 AND t.kind_id<7
SELECT COUNT(*) FROM title t WHERE t.kind_id<3 AND t.production_year<1967
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=3 AND t.kind_id<7 AND t.production_year=2011
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id=948 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=4
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1142555
SELECT COUNT(*) FROM title t WHERE t.kind_id=1 AND t.production_year=2010
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<352350
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id>1 AND t.production_year<2003
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>18302 AND t.kind_id<7 AND t.production_year>2005
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id<75508
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id<73804 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>29088 AND t.kind_id=7 AND t.production_year=2002
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<16 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>16 AND mi_idx.info_type_id<101 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=2792704 AND t.kind_id=7 AND t.production_year<2005
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2157677
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<7 AND t.production_year>2008
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>358323 AND ci.role_id=1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1639083
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3428218
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<9910
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<997596 AND ci.role_id=10
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1350122 AND ci.role_id=10
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=1013
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id=935267 AND mi.info_type_id=16 AND t.kind_id>4 AND t.production_year<2001
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>15116 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<3
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=2394
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=38613
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<169
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<16 AND t.kind_id<7 AND t.production_year<2008
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>29121
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=5867
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1033058 AND ci.role_id>4
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id=16483 AND mc.company_type_id<2 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>3 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2172013 AND t.kind_id<4 AND t.production_year<1929
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<664
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<19 AND t.kind_id=7 AND t.production_year=1971
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<773743 AND t.kind_id>3 AND t.production_year>2001
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id<101 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>190061 AND mc.company_type_id>1 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<8 AND t.kind_id<7 AND t.production_year>2010
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>211605
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id=81472 AND mi.info_type_id<8 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=1 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=39431
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>120874
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=7 AND t.production_year<1975
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id>48
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>7546
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1637534
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id=486 AND t.kind_id>2
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=55201 AND t.kind_id=7 AND t.production_year>1992
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<18 AND mk.keyword_id>5261
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<559 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>86128 AND t.kind_id<7 AND t.production_year>2000
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<2809195 AND ci.role_id=2 AND mc.company_id<72366 AND mc.company_type_id<2 AND t.kind_id=3 AND t.production_year>2001
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>152484
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>1596
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id>8
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8 AND ci.person_id<729910
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3042538 AND ci.role_id>1
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>171 AND mc.company_type_id<2 AND ci.person_id>3433913 AND t.kind_id<4
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>1414566 AND mk.keyword_id=2867 AND t.kind_id=1 AND t.production_year<2007
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<82046 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id=382 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>101 AND t.kind_id=1 AND t.production_year=1989
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND ci.person_id>2746203 AND t.kind_id>2
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>160
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3278257
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year<1999
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=102
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id<202772
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=49249 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=5542
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>100 AND mk.keyword_id>1776 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>16 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id<100
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<1284
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id=94847 AND mc.company_type_id>1 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>1301595 AND ci.role_id=10
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>666 AND t.kind_id=1 AND t.production_year=1963
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>99 AND t.kind_id=4
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id<101 AND mk.keyword_id=12147
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<155856
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3834689 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>1382
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3531097
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3471456 AND ci.role_id=10
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id<4 AND t.kind_id=7 AND t.production_year=1996
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>5768
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<3577544 AND ci.role_id=1 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id<16
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>615831
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3182994
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=2707
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=3452
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>20857
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id=1142636 AND ci.role_id<2 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>15
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1859279 AND ci.role_id>10
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>39071 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>101 AND mi.info_type_id<3
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>329356
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.kind_id>1 AND t.production_year>2003
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>94 AND t.kind_id<7 AND t.production_year>2012
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>87294 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id=11203 AND mc.company_type_id=1 AND mk.keyword_id>16264
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>8 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=1 AND t.production_year=2003
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1901085 AND ci.role_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND ci.person_id<20804 AND ci.role_id=2 AND t.kind_id<4
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3561511
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>32881
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=494
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id>44517 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<311905 AND ci.role_id=8
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id<58435
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>138227 AND mc.company_type_id>1 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>23486 AND mc.company_type_id=1 AND mk.keyword_id<465 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<12961 AND mc.company_type_id=2 AND t.kind_id=1 AND t.production_year<2010
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=2375 AND t.kind_id=7 AND t.production_year=2003
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>835163 AND ci.role_id=2 AND mk.keyword_id=39905 AND t.kind_id>3 AND t.production_year>2006
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<158418 AND mi.info_type_id=8 AND t.kind_id=1 AND t.production_year>2001
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=10510
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<1848590 AND ci.role_id>3 AND mi_idx.info_type_id=99
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1552243
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<652
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3 AND mi_idx.info_type_id<101 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<708998 AND t.kind_id=3 AND t.production_year>2013
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id=2879493
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<9707 AND t.kind_id<7 AND t.production_year=2012
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2563258
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id=3747 AND mc.company_type_id=1 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<322086
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<820
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>4
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1351205 AND ci.role_id>1
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id=2990405 AND ci.role_id>3 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>6
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id<182536
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<4 AND t.production_year=2009
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id=3
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<151181
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>1642114 AND t.kind_id>1 AND t.production_year=2004
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<3
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<4617
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>401 AND mk.keyword_id<2879
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year=1912
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id=115 AND mc.company_type_id<2 AND mi.info_type_id<2 AND t.kind_id<2 AND t.production_year>1970
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=18 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3092395 AND ci.role_id>2 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=2623105 AND ci.role_id<3
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year>1913
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id>1869 AND mc.company_type_id>1 AND t.kind_id<7 AND t.production_year<1973
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>2488 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1286381 AND ci.role_id<4
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND mi.info_type_id<2 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>98 AND t.kind_id>2
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id>16 AND t.kind_id<7 AND t.production_year<2004
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<370733
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>281416
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>80527 AND t.kind_id<7 AND t.production_year<1987
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=6
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>2002
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id>878330 AND t.kind_id>2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=15345
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1677301 AND ci.role_id<6 AND t.kind_id>4
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id<6363 AND t.kind_id=7 AND t.production_year=2006
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1462486 AND ci.role_id<2
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id=2329 AND mc.company_type_id<2 AND t.kind_id=3
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>2 AND t.kind_id<3
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>1434 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<10100
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>3 AND t.kind_id=7 AND t.production_year>2008
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>4023 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=48 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=826
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<120576
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<1714 AND mc.company_type_id=1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<292475 AND t.kind_id<7 AND t.production_year=2008
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=24091
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND mi.info_type_id>16 AND t.kind_id>1 AND t.production_year=1975
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id=1900367 AND ci.role_id<6 AND t.kind_id<7
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>2013
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<2674161 AND ci.role_id>3
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=4 AND ci.person_id<511353 AND ci.role_id>1 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>99
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8 AND mi_idx.info_type_id=100
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<89622
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=99
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>429415 AND ci.role_id>2
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.kind_id>3
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>83238
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<13543
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>570455 AND ci.role_id<8 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id>16
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=7731
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id>4
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<75446 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=1074
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2889767
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<14947 AND mc.company_type_id>1
SELECT COUNT(*) FROM title t WHERE t.kind_id<4 AND t.production_year=1998
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id>536
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<8
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year=2010
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2930976 AND ci.role_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id>7820 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1398163 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<10584 AND mc.company_type_id=1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>503990 AND t.kind_id<2
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id=1 AND t.production_year=2006
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id=3464
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>11631 AND mk.keyword_id=2882 AND t.kind_id<2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=7498
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=1 AND t.production_year<1977
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id<3
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>1957
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>10294 AND mk.keyword_id=234 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=2782152 AND ci.role_id=4 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<1244102 AND ci.role_id>2 AND mi.info_type_id<4 AND t.kind_id=7 AND t.production_year>1948
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=7 AND t.kind_id>2
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id<4928 AND t.kind_id<4
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<4 AND mk.keyword_id>70044
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id>326
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>156726 AND mc.company_type_id=2 AND mi.info_type_id=3
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND mi.info_type_id=16
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<98 AND t.kind_id=7 AND t.production_year=2005
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<395501 AND ci.role_id=2 AND t.kind_id<7 AND t.production_year<2010
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=7 AND t.production_year>2010
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<8424
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>16
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<3243738 AND t.kind_id<7 AND t.production_year>1969
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id<115 AND t.kind_id=2
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id>23620 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<106
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1904832
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>2438920 AND ci.role_id>2 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.kind_id=7 AND t.production_year>2011
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>3316421 AND t.kind_id<7 AND t.production_year<2007
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=3747
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=160 AND mc.company_type_id=1 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=2 AND t.kind_id=7 AND t.production_year>2010
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id=59203 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id>1267
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>2
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=18238 AND t.kind_id=1
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year=2013
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<2
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<3135 AND mc.company_type_id=2 AND mi.info_type_id<4 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<160
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2003032
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id=7 AND t.production_year>2009
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year=1992
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<222675 AND t.kind_id<7 AND t.production_year=1972
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id=302 AND mk.keyword_id>32560
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3449010
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id<7 AND t.production_year<1999
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>200 AND mk.keyword_id<382
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<669 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<179666
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<683
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=3227766 AND ci.role_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=232
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id=428 AND mc.company_type_id<2 AND ci.person_id>290454 AND t.kind_id>3 AND t.production_year>2004
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<3439776 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>1 AND t.production_year>1911
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>50406
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<956469 AND ci.role_id>1 AND mk.keyword_id>121
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>1067 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<16821 AND mc.company_type_id<2 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>7 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<16 AND mc.company_id<160 AND t.kind_id<3
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1632410
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>528236 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<1510238 AND mk.keyword_id<18649 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>48921 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=99 AND t.kind_id<7 AND t.production_year>0
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>145 AND mc.company_type_id=2 AND mi_idx.info_type_id<101 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1434953 AND t.kind_id>3
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id<3089 AND t.kind_id=7
SELECT COUNT(*) FROM title t WHERE t.kind_id<4
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND t.kind_id=7 AND t.production_year<1998
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<105091
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<9 AND mk.keyword_id=75460
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2547077 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=5443
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.kind_id>1 AND t.production_year>2001
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<98 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>158 AND mc.company_type_id=2 AND t.kind_id=7 AND t.production_year>2012
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>3
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=1 AND t.production_year>2012
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=100 AND t.kind_id<7 AND t.production_year=2004
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=85
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>6924 AND mc.company_type_id=2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2052053
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=1 AND t.production_year<2011
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<807 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id=107 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.kind_id=7 AND t.production_year=1944
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year<2011
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>17
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=7 AND t.production_year>2011
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>158334 AND mk.keyword_id<9236 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<904
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=275
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND ci.person_id>43958 AND ci.role_id<3
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=7 AND t.production_year=2004
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3268042 AND ci.role_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3448403 AND ci.role_id>7
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<1462855 AND mi_idx.info_type_id>101
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>10922 AND t.kind_id<7 AND t.production_year>1966
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>21498
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2223776 AND ci.role_id>2
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>3 AND mk.keyword_id=1686 AND t.kind_id<3 AND t.production_year<2011
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id<4 AND t.production_year<2011
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>30717 AND mc.company_type_id<2 AND mk.keyword_id>901
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=712
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>0
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<14326
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=22890
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.kind_id>1 AND t.production_year>1912
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2355687
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year<1992
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<34 AND mc.company_type_id=2 AND mi_idx.info_type_id<101
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1435303
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<1597779 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>16 AND mc.company_id=6 AND mc.company_type_id=1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id>10212
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>43488
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>15405 AND mc.company_type_id<2 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>10068
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id=424 AND mk.keyword_id<3627 AND t.kind_id=7 AND t.production_year>1922
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<29078 AND mc.company_type_id=1 AND t.kind_id=7 AND t.production_year=2000
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3906369
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<9104 AND t.kind_id>1 AND t.production_year=1971
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>13625 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND t.kind_id<7 AND t.production_year=1999
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id<7
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>2159923 AND ci.role_id<4
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=8 AND t.kind_id>1 AND t.production_year=1990
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>679
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>32922
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=4836
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1023229
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>562
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id>1 AND t.production_year>2007
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id<100
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<5623
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>6012 AND t.kind_id<4 AND t.production_year<2011
SELECT COUNT(*) FROM title t WHERE t.kind_id=1 AND t.production_year<1904
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id=1885410 AND t.kind_id=7 AND t.production_year>1974
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id<101 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=697956 AND ci.role_id<10
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id<6 AND mc.company_type_id>1 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1631964 AND ci.role_id>3 AND t.kind_id=1
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year>2011
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id=7 AND t.production_year=2003
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>17 AND mk.keyword_id=1127 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=796885 AND t.kind_id<7 AND t.production_year<2007
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>1657
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id<2 AND t.production_year=2013
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id=7 AND t.production_year>2008
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<21553
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id>47838 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1966315 AND ci.role_id>10
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<6158 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<101595
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<793
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=172637 AND ci.role_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id=7 AND t.kind_id=7 AND t.production_year=1983
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>8
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id>1 AND t.production_year=1971
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2561921
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<65419 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<11146
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<17455 AND t.kind_id>4
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<3 AND t.production_year<1985
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=100 AND mk.keyword_id>13436 AND t.kind_id<7 AND t.production_year<2004
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year=2008
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<605976
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1844019
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=4847
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=3 AND t.production_year>2009
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=305 AND t.kind_id<7 AND t.production_year=2012
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id>12338 AND t.kind_id>1 AND t.production_year<2009
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>63651
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year>2004
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>92 AND t.kind_id=1 AND t.production_year=1965
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id<22111
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1926695
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<13
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id<5489 AND t.kind_id<7 AND t.production_year>1994
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1291343 AND ci.role_id=10
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<14056 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.production_year=1911
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2474124
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>80750 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id<3
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16 AND t.kind_id=7 AND t.production_year<2008
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id=7 AND t.production_year=1994
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<844551
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<133097 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<2515823
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<7 AND t.production_year>1967
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1120142
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>4043057 AND t.kind_id>1 AND t.production_year>1996
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id<4 AND t.production_year>2006
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<83703 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>3545
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id=3044905 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>16 AND t.kind_id<2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2802502 AND ci.role_id<4
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id<13 AND t.kind_id>2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=2269
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2821290 AND ci.role_id=1 AND t.kind_id=4 AND t.production_year>1960
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id=63537
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<3796559
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<442718 AND t.kind_id>4 AND t.production_year<1962
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<239 AND mc.company_type_id=2 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<11713
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id>856552 AND mc.company_id=1256 AND mc.company_type_id=1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1018677
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year<1910
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2636444 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>8 AND mk.keyword_id>11577 AND t.kind_id=7 AND t.production_year<2011
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<2218
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>7 AND mk.keyword_id=28336
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1841593
SELECT COUNT(*) FROM title t WHERE t.kind_id<2 AND t.production_year=2011
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>24897 AND t.kind_id<7 AND t.production_year<2001
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<14056
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2508315
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<2148411 AND ci.role_id=8 AND mc.company_id>191 AND mc.company_type_id>1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2114088 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>43569
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=7 AND t.production_year<1988
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<12093
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.kind_id=1 AND t.production_year>1995
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>785499
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2568220 AND t.kind_id=2
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id<200485
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id>42380 AND t.kind_id<7 AND t.production_year>1960
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=428
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1801044 AND ci.role_id>5
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=4 AND t.production_year<2005
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>99 AND mk.keyword_id>17728 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year>2009
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<2 AND mc.company_id>858 AND t.kind_id=7 AND t.production_year<2004
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id<2 AND t.production_year>2012
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>8754 AND mc.company_type_id<2 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.kind_id>1 AND t.production_year=2012
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND t.kind_id<4
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>71050 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<11146 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=153676
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3853835
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>101 AND mk.keyword_id<20344 AND t.kind_id<7 AND t.production_year=1954
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2323717 AND ci.role_id<6
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.kind_id=7 AND t.production_year=2003
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=927 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id>100 AND t.kind_id=1 AND t.production_year>1958
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year<1993
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id<2 AND mk.keyword_id=167 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>29335
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=673119 AND ci.role_id=4
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>1 AND t.kind_id>4
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id>99 AND t.kind_id=7 AND t.production_year=2005
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND mc.company_id=19869
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND t.kind_id=1 AND t.production_year=1993
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>5376
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id<1215
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<1526 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND mc.company_id=18262
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=2208
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<335
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=16264 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>1649 AND mc.company_type_id=1 AND t.kind_id>1 AND t.production_year>2001
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>394 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id>14819 AND t.kind_id=1 AND t.production_year=1990
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>1 AND t.kind_id>1 AND t.production_year=2006
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<1653102
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id>571
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>2807274 AND ci.role_id=10 AND mi.info_type_id=16 AND t.kind_id<7
SELECT COUNT(*) FROM title t WHERE t.kind_id=1 AND t.production_year<2009
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id=3501 AND mc.company_type_id=1 AND mi.info_type_id>7 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id<6321 AND t.kind_id=2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id<7 AND t.production_year>1942
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2769391
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=2405
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=2 AND t.production_year=1973
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2752751 AND ci.role_id<2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2524329 AND ci.role_id=1
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>15 AND ci.person_id>2948508 AND ci.role_id=3 AND t.kind_id=7 AND t.production_year=2011
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<1742
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id=82751 AND t.kind_id=2 AND t.production_year<1996
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<2805
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3228957 AND ci.role_id>10
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>2009
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=6
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND ci.person_id>916099 AND t.kind_id>6 AND t.production_year>2003
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=323214 AND ci.role_id>3 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<2294258 AND ci.role_id=2 AND mc.company_id>1700 AND mc.company_type_id=1
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<2280472 AND ci.role_id>1 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>105269
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=12877 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<3 AND t.kind_id=1 AND t.production_year=2005
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>21308
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year>1984
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=107 AND t.kind_id=1 AND t.production_year<2005
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=99 AND t.kind_id=7 AND t.production_year>2012
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<423117
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=1374 AND mc.company_type_id>1 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=28495
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id>3668 AND t.kind_id=7 AND t.production_year<2009
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<3127266 AND t.kind_id=1 AND t.production_year<2011
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>16 AND mk.keyword_id<6326 AND t.kind_id>2 AND t.production_year>1987
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year=2005
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=99 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2889663 AND ci.role_id>3
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id=91464 AND mk.keyword_id>4893 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>14682
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>453
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<1906
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2118933
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id>16994 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>101 AND mk.keyword_id=103 AND t.kind_id=1 AND t.production_year<1999
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=2375 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.kind_id>1 AND t.production_year=2004
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year<2007
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id<7 AND t.production_year=2009
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>1484 AND mc.company_type_id=2 AND t.kind_id<7 AND t.production_year>1978
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND mc.company_id=16432 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>920362 AND ci.role_id=1 AND t.kind_id=1 AND t.production_year<1997
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=7 AND mc.company_id=88243
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<8 AND mc.company_id<4292 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND mc.company_id<65841 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND t.kind_id=7 AND t.production_year>2004
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>463806 AND ci.role_id<4
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>32998
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id<3347501 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=2215 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<8
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>11156
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<1483 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND mk.keyword_id=20319
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id>62
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1010362 AND ci.role_id>1
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>2008
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=18486
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND mc.company_id>8142 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<58
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>21715 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>24621
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>45522
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id<100 AND t.kind_id>2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<24459
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<1427
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<160 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>16 AND mi_idx.info_type_id=100 AND t.kind_id>1 AND t.production_year=1975
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<18534 AND t.kind_id=7 AND t.production_year=2009
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id=7 AND t.production_year<2007
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>57775 AND mc.company_type_id=2 AND mi.info_type_id<95 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<1139906 AND ci.role_id>4 AND mk.keyword_id<13641 AND t.kind_id=7 AND t.production_year<1998
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=1309
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id<373
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<27357 AND ci.role_id=10
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>719932 AND ci.role_id<10 AND mk.keyword_id<17652
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id<7 AND t.production_year<1994
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=3
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>24829
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1103225 AND ci.role_id=4
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>419627
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>2684549 AND ci.role_id>10 AND mi.info_type_id<94 AND t.kind_id<7 AND t.production_year>0
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>150355
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>18788
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<112323 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id>4333 AND t.kind_id>2 AND t.production_year<1983
SELECT COUNT(*) FROM title t WHERE t.kind_id<3 AND t.production_year<2009
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>6 AND mi.info_type_id=1 AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>42755 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8 AND mc.company_id=708 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>1411
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>1633
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>4188 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id=16 AND ci.person_id<3473415 AND ci.role_id>1 AND t.kind_id=7 AND t.production_year=2007
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id=1886 AND t.kind_id>3 AND t.production_year<2010
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>2554
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id<4802
SELECT COUNT(*) FROM title t WHERE t.kind_id=1 AND t.production_year=1988
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>8 AND mk.keyword_id<335
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=1 AND t.production_year=2013
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>1695 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>17939 AND mi.info_type_id>16 AND t.kind_id=4 AND t.production_year>2009
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND ci.person_id<434371 AND ci.role_id>10
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=25083
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id>27823
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>27168
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>99 AND mk.keyword_id<14460 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id>3
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<521 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>596
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=174670
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id=6 AND ci.person_id<623698
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id=666 AND t.kind_id>3 AND t.production_year>1958
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>69 AND t.kind_id=7 AND t.production_year<1974
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>100 AND t.kind_id>1 AND t.production_year=1997
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>38236 AND t.kind_id=1 AND t.production_year<2013
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1012001
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<16 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=26576
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>4 AND mc.company_id<6893 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<48698 AND t.kind_id=1 AND t.production_year=2004
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id<101 AND t.kind_id>1 AND t.production_year>1996
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2706565
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>69560 AND t.kind_id=7 AND t.production_year>2009
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=687 AND mc.company_type_id<2 AND t.kind_id<7 AND t.production_year>2010
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<690664 AND ci.role_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year>1953
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<4
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id<160 AND t.kind_id=1 AND t.production_year>2010
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>464 AND mc.company_type_id=1 AND mk.keyword_id>902
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<19968
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3746009
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>28339
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=3 AND t.production_year>2011
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id>14577 AND t.kind_id=2 AND t.production_year>1997
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2258148 AND ci.role_id>10
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1105744
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1589592
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>141
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<8185
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<1709
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id>6736 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<16 AND t.kind_id<3
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=5529 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>957 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>7 AND ci.person_id=173263 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>13351
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id>1 AND t.production_year>2010
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3193309
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=2870
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year>2003
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=12
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id=102 AND mi.info_type_id<15 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<2614 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<2451
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id<100 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>18790 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<19 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<94745
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16 AND mk.keyword_id<29318 AND t.kind_id<7 AND t.production_year<2012
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND ci.person_id=1495180 AND ci.role_id=1 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id>8
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id>66251 AND t.kind_id=7 AND t.production_year>2012
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id=16 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id=11203 AND mc.company_type_id=2 AND mk.keyword_id<772 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<121 AND t.kind_id=7 AND t.production_year<2006
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id=2 AND t.kind_id=7 AND t.production_year>1938
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id=31615 AND mc.company_type_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2024377
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<846
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id<137 AND t.kind_id=3
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=998196 AND ci.role_id=1 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.kind_id=7 AND t.production_year=2004
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>2890108 AND ci.role_id>3
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<73096 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id<7 AND t.production_year>2005
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id<3210195 AND ci.role_id=3
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>31051
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<4 AND t.production_year>1964
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<2976834 AND mk.keyword_id<1876
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id=13115
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>3801885
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<11872
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id=549164 AND t.kind_id<7 AND t.production_year<1975
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>10018
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>39855
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<16 AND t.kind_id<7 AND t.production_year<1972
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id>4
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=7
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND t.kind_id>2 AND t.production_year>2012
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>2917295 AND ci.role_id>1 AND t.kind_id>3
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<11244 AND mi_idx.info_type_id=99
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=3 AND t.production_year<2013
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<2860 AND t.kind_id<7 AND t.production_year<2011
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>968994 AND ci.role_id=2 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=37155 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1424446 AND t.kind_id=3 AND t.production_year>1991
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id>3
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<25813 AND mc.company_type_id>1 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>165679 AND ci.role_id<3 AND mi_idx.info_type_id<101
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND mk.keyword_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id=91100 AND t.kind_id<3
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1850349
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year=2007
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=104927
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>4010170
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>359
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<503511 AND t.kind_id<7 AND t.production_year>2012
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<3346
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=8 AND mk.keyword_id=26123
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>3664293 AND mi.info_type_id>6 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id<100507 AND t.kind_id=7 AND t.production_year<2007
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year>2013
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id<395
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<8917 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=77195
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=28011 AND mc.company_type_id=1 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>13816 AND mc.company_type_id>1 AND t.kind_id=7 AND t.production_year>2011
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year>1971
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8 AND ci.person_id>972625 AND ci.role_id=1
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<1504949 AND ci.role_id=4 AND mc.company_id<19 AND mc.company_type_id>1 AND t.kind_id=7 AND t.production_year=1985
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>1 AND t.production_year>2003
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<4 AND ci.person_id>2924897
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>2875586 AND mi_idx.info_type_id=101 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>5231
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<168880 AND ci.role_id=1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.kind_id>1 AND t.production_year=2003
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id=6
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2904955 AND ci.role_id>1 AND t.kind_id=7 AND t.production_year>2006
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND t.kind_id<3
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<2053
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<1828
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id>1780 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>93
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<297133 AND mc.company_id<6 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=103341 AND ci.role_id=1 AND t.kind_id<7 AND t.production_year<2005
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<13 AND t.kind_id=7 AND t.production_year<1968
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>2225978 AND ci.role_id<4
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<1732 AND t.kind_id=7 AND t.production_year<1985
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<11151 AND mc.company_type_id=2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1406434
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<1747690 AND mc.company_id>152648
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id>1 AND t.production_year=1965
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<3247 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND mc.company_id>166
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id>1 AND t.production_year=2008
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND t.kind_id=7 AND t.production_year<1992
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<97307
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<6 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=5 AND ci.person_id>2974604 AND t.kind_id=7 AND t.production_year<2007
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.kind_id=7 AND t.production_year>1934
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2116514 AND t.kind_id<4
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=99 AND t.kind_id=7 AND t.production_year>0
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=112301 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id=2168798 AND ci.role_id>1 AND t.kind_id>6
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id<7 AND t.production_year=2008
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND mi.info_type_id=16
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id>1 AND t.production_year<2007
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<11141
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>411
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2018707
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year>1993
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<530 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=100 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>20329
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=12807
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>939427
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>34
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id<2616 AND t.kind_id<7 AND t.production_year=1987
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<66
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<1230 AND t.kind_id<7 AND t.production_year<2001
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=87983
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year<2000
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>101 AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=2112 AND t.kind_id<2
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<1074
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>8142
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>8814
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id<7 AND t.production_year>2007
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<14212
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<22427
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND mk.keyword_id=16264
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>317 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2261178 AND ci.role_id>2
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id<27
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2734249
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<2505360 AND mi_idx.info_type_id<101
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<7514
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1989774
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1350232
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id<3 AND t.kind_id=7 AND t.production_year<2010
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<7 AND ci.person_id=3278672 AND ci.role_id<10
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>52 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=7 AND t.production_year=2010
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>598 AND mc.company_type_id<2 AND t.kind_id=7 AND t.production_year<2006
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id=107
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=99 AND mk.keyword_id>382
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id=3 AND t.production_year=1970
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>129200 AND ci.role_id=3
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>12024
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id>20606
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=7 AND t.kind_id>2 AND t.production_year>2003
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<6 AND t.production_year>2012
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<12922
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>135028 AND ci.person_id>2098413 AND ci.role_id>6 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>12929
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>596
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id=31716 AND mc.company_type_id=2 AND ci.person_id>334911
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<2393 AND t.kind_id=7 AND t.production_year>1910
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id>1 AND t.production_year>1996
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>101
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>895
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>5898 AND t.kind_id=1 AND t.production_year=2005
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id<67405
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=3255290
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3211768
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>1240
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>6 AND mk.keyword_id=5604
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=1283 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2246424
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=2158 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>4625
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id>101
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=53351 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id<11163 AND mc.company_type_id=1 AND t.kind_id=2
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND t.kind_id=7 AND t.production_year>1971
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year<1999
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=3
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year=2009
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<1382
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>73820 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>5973
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year>1979
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id<86822 AND mc.company_type_id=1 AND ci.person_id>2339607 AND ci.role_id<4
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2858622
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=904 AND mc.company_type_id=2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2920850
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<73511 AND t.kind_id=2 AND t.production_year>2010
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id=2127402 AND mk.keyword_id>742 AND t.kind_id<2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3060788 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=15 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=67840 AND ci.role_id<11 AND mk.keyword_id<6962 AND t.kind_id<7 AND t.production_year<2013
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id<16 AND mk.keyword_id=7395 AND t.kind_id=1 AND t.production_year=2002
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=47
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>4629
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=50384 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3708409 AND ci.role_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year<1967
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<71756 AND ci.role_id<2 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>57781
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>112252 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=16 AND t.kind_id=1 AND t.production_year=2012
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<623943 AND ci.role_id=4 AND mi_idx.info_type_id>99
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id<7 AND t.production_year=1997
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>1295423 AND mi.info_type_id=4
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year>1973
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=7 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<705933 AND ci.role_id=11
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=23579
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id=6024
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year<2000
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=498
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<88639
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2316330 AND t.kind_id<7
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year=2004
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2497502 AND ci.role_id<10 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=8610 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<8 AND mc.company_id<16954
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id<22346 AND t.kind_id<7 AND t.production_year=1960
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<3951905 AND ci.role_id>4
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id=11252 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>101 AND mk.keyword_id<8479 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<135358 AND ci.role_id=3
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id=71146 AND ci.person_id>1868283 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1223318 AND ci.role_id>1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>4002533
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id<2058984 AND ci.role_id=2
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id>466
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2174766
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=79912
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<105 AND t.kind_id<2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>432518 AND ci.role_id<10 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<11172
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3892046 AND ci.role_id=4
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id=222493 AND ci.role_id<4 AND t.kind_id=2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year<2012
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND mi.info_type_id=1 AND t.kind_id=7 AND t.production_year=2001
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2978454 AND ci.role_id>1 AND t.kind_id<7 AND t.production_year>0
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>101768
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=10205
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1316857 AND ci.role_id=2
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>6 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2629068 AND ci.role_id>3
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=8833
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1924561
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<24830 AND t.kind_id>3
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=5352
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=1 AND t.production_year>2011
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>3256
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id<4
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=3612430
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year>0
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<2022628 AND ci.role_id<10 AND t.kind_id>1 AND t.production_year>2008
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>264 AND mc.company_type_id=2 AND t.kind_id<4
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>2926 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>4000049
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>16 AND mk.keyword_id>7078 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=3649957 AND ci.role_id>3
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>2626441
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<3040499 AND ci.role_id<3 AND t.kind_id=3 AND t.production_year>2009
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>13791
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<1751
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=24528
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id>3636 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=875
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id>17 AND t.kind_id=1 AND t.production_year>1989
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id>305138
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=34 AND mc.company_type_id>1 AND t.kind_id<7 AND t.production_year<2012
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>1925831 AND ci.role_id>3 AND mi.info_type_id<3 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<6 AND t.kind_id=7 AND t.production_year=2010
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<342171 AND ci.role_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>11149 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id>2
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=1 AND t.production_year>2002
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year<2001
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=13435
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>611438 AND ci.role_id<4 AND t.kind_id=7 AND t.production_year<1963
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<20245 AND t.kind_id=7 AND t.production_year>2005
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id=1742 AND t.kind_id=1 AND t.production_year<1983
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>164 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=225
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<16 AND t.kind_id=7 AND t.production_year=2013
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<88453 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id>38955 AND t.kind_id>1 AND t.production_year=2003
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1521014 AND t.kind_id=2 AND t.production_year<1993
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>16 AND mk.keyword_id<3921 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<5
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id>1 AND t.production_year<2012
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND ci.person_id>1259254 AND ci.role_id=1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<3192824 AND mk.keyword_id>498
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<15 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id=4
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<2609 AND mi_idx.info_type_id>101
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>750 AND t.kind_id=7 AND t.production_year>2002
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1187146 AND ci.role_id>2 AND t.kind_id=7 AND t.production_year<1950
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND mc.company_id<71740 AND t.kind_id=7 AND t.production_year>2007
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2295888
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND mi.info_type_id=3 AND t.kind_id>1 AND t.production_year=1995
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=820869
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id>100 AND t.kind_id<7 AND t.production_year=1993
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id<16
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id<4595 AND t.kind_id<7 AND t.production_year=2004
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND t.kind_id>1 AND t.production_year=1977
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=4962
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>2561
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<51137
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<21416
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=3 AND ci.person_id<885994 AND ci.role_id>2 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1360973
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=2273
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=1699
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id=160 AND mc.company_type_id=2 AND mi_idx.info_type_id<101 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2355788 AND t.kind_id=7 AND t.production_year=2010
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2731675 AND ci.role_id=10
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=223666
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>83622
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>192226 AND ci.role_id<2
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND t.kind_id>4 AND t.production_year>2007
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=7 AND t.production_year<2006
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=870 AND t.kind_id<7 AND t.production_year>1996
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3098462
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<103292 AND ci.role_id<10 AND t.kind_id<7 AND t.production_year<2006
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3084143 AND ci.role_id<10
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>106 AND ci.person_id>2489206
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=7 AND t.production_year=1984
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1357914
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id>5231 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<80760
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>59435
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<918
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id<7525 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=11463
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id=16264
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2079250
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>3 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=480 AND mc.company_type_id<2 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id<1496494 AND t.kind_id>1 AND t.production_year=1979
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<1951
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<14419
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id<69493
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<2479558 AND t.kind_id<7 AND t.production_year=2000
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<103989
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year<1910
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=399
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<102
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year<2013
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id<6 AND mi_idx.info_type_id>100 AND t.kind_id=7 AND t.production_year=2012
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<9
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<258216 AND mk.keyword_id<21711
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>341 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>4 AND t.kind_id<7 AND t.production_year<1999
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1696902 AND ci.role_id=4
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<1820 AND t.kind_id=7 AND t.production_year>1990
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1631607
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=500
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id=3 AND t.production_year<1979
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>649446
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=100 AND t.kind_id<4 AND t.production_year=1970
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=495
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id>43148
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year=1981
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=78613
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>24528 AND ci.role_id<2
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=7 AND t.kind_id<7 AND t.production_year>1969
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>1985
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>2143 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>17619
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1943017 AND t.kind_id=3
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<11149
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>2885901
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<13992 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>101 AND t.kind_id<4 AND t.production_year<2001
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1348446
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>7890
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id<101 AND t.kind_id=1 AND t.production_year<2002
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>186194 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>100 AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=718
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id<11149 AND mc.company_type_id>1 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2180573 AND ci.role_id>2
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id=2028539
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=1741
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1824387
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16 AND mk.keyword_id<6901 AND t.kind_id<7 AND t.production_year>2009
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1011448 AND ci.role_id=4
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>1533479 AND ci.role_id<2 AND mi_idx.info_type_id>100
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2570698
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>88998 AND mc.company_type_id=2 AND mk.keyword_id<4449
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>1698383 AND ci.role_id>1
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id=2707827 AND mi.info_type_id>16 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<40224 AND mc.company_type_id=1 AND mi.info_type_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3153511
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>15 AND t.kind_id=3 AND t.production_year=2009
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<3825488 AND ci.role_id=6
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>387
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=9664 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.kind_id>1 AND t.production_year=1972
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1075387 AND ci.role_id=2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2770239
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<34971
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>11894 AND t.kind_id=4 AND t.production_year>1989
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=100 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=359
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id>2001602
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=5254
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND mc.company_id>140288 AND t.kind_id>2
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<56980 AND t.kind_id<7 AND t.production_year>1995
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<18949 AND mk.keyword_id<20805 AND t.kind_id=7 AND t.production_year<2002
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>1610
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year>2012
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<63155
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>56759 AND mc.company_type_id=2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1824978 AND ci.role_id<9
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id=73977 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id<263 AND t.kind_id<4 AND t.production_year>1996
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=3111
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id<103 AND mk.keyword_id<34313 AND t.kind_id>1 AND t.production_year=1960
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<71614 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>99 AND mk.keyword_id<335
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1875936 AND t.kind_id=7 AND t.production_year<2010
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.kind_id=7 AND t.production_year<2008
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=1 AND t.production_year<2013
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id=9328
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<6 AND mc.company_type_id<2 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>20901 AND mc.company_type_id>1 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND mi.info_type_id=16 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id=45825 AND t.kind_id=1 AND t.production_year<1998
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1650220 AND ci.role_id=10 AND t.kind_id>4
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>596 AND mc.company_type_id>1 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<50979
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>1002168 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>8 AND t.kind_id=7 AND t.production_year=1994
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>6 AND mc.company_type_id<2 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id<14291 AND ci.person_id<800495 AND t.kind_id>1
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year=1963
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2108364 AND ci.role_id=2
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND t.kind_id=7 AND t.production_year<2008
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<139 AND mi_idx.info_type_id<100
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id=9499
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<7 AND t.production_year<2007
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<48348
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>6 AND t.kind_id<7 AND t.production_year<2004
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<196
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<11278
SELECT COUNT(*) FROM title t WHERE t.kind_id<3 AND t.production_year>1989
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id>100
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>6 AND t.kind_id=2 AND t.production_year<2010
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<233512 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id<106 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<37261 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3790666
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.production_year=2004
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1692347 AND ci.role_id>3
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=56603
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id<79800 AND t.kind_id=7 AND t.production_year=1999
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id=3636 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=763438
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>14550
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id>7679 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id=106 AND t.kind_id>1 AND t.production_year=1989
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND ci.person_id<788687 AND ci.role_id>9 AND t.kind_id=7 AND t.production_year<2012
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>4660
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=1 AND t.production_year=1909
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>305
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id=15
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>5 AND mc.company_id<160 AND mc.company_type_id=1 AND t.kind_id>1 AND t.production_year>2013
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<270
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>15 AND t.kind_id>1 AND t.production_year<2012
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>82933
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<952644 AND t.kind_id<2 AND t.production_year<2000
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<1562753 AND ci.role_id<2 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=2919739 AND t.kind_id>4
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id<101 AND t.kind_id=4 AND t.production_year>2003
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<75081 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<5898
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1216132
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>5262
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<122811
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<1486 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>17757 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<37808
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=158307
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>2845
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.kind_id<7 AND t.production_year=1988
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id=13
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND ci.person_id>2354017 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=7589
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id=149223 AND ci.role_id<8
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>101 AND mc.company_id>12684 AND mc.company_type_id<2 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id=7 AND t.production_year<2007
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<5902 AND t.kind_id>1 AND t.production_year=1989
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<24607 AND t.kind_id=7 AND t.production_year=2005
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<530 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1722929
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<247349 AND mi_idx.info_type_id=100 AND t.kind_id<7 AND t.production_year<1998
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>1 AND t.production_year=1975
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>17 AND mc.company_id=6285 AND t.kind_id<7 AND t.production_year<2011
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id<1150
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>13015
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>1713 AND ci.person_id=946460 AND ci.role_id<10
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>7811 AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>12489
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=319
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<40321
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year=2005
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id=992 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=1724
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=118918
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>228628 AND t.kind_id<7 AND t.production_year>2005
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>85125
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>5 AND t.kind_id=7 AND t.production_year>2012
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id<81416 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1617115 AND ci.role_id>1
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id<4 AND t.kind_id<7 AND t.production_year<1998
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<1828 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id<7 AND t.production_year>2012
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id=1 AND t.production_year<1903
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id>506132 AND mi.info_type_id=2 AND t.kind_id=7 AND t.production_year=1999
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1394329
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id>1 AND t.production_year>2001
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=347
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>157298
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id>1 AND t.production_year=2008
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1845692
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id>856 AND t.kind_id=7 AND t.production_year<2001
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2206631
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND t.kind_id=2 AND t.production_year<2007
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<527511 AND ci.role_id<3
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=50372
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year=2009
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id>556463 AND ci.role_id<4 AND t.kind_id<2 AND t.production_year=1990
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id<7 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2655351 AND ci.role_id=3
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<781853
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>1162094 AND t.kind_id<3
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<12823
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2761777 AND ci.role_id<10
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id<8 AND mi_idx.info_type_id<101
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id=2123251 AND mc.company_id<480 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>16 AND t.kind_id<4 AND t.production_year=2007
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=394 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>555
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id<382 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<2456874 AND mi_idx.info_type_id>100 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>8719
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=32970
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<15 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND ci.person_id>529316 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>18
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<3572624 AND t.kind_id>1 AND t.production_year>2002
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3099930
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<8 AND mc.company_id=18804
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1431499
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<1494064 AND ci.role_id=2 AND mi_idx.info_type_id>99
SELECT COUNT(*) FROM title t WHERE t.kind_id<2 AND t.production_year>2005
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id=3755 AND mc.company_type_id=2 AND ci.person_id>119053 AND ci.role_id=2 AND t.kind_id<7 AND t.production_year<2005
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id=26960 AND mc.company_type_id=1 AND t.kind_id=1 AND t.production_year<2009
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2779201 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=84381
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<3212417 AND mi.info_type_id=7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<480
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>99 AND mk.keyword_id>373 AND t.kind_id>3
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id<3921
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id<15 AND ci.person_id=175411 AND t.kind_id<4 AND t.production_year<2012
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<11219 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=7 AND ci.person_id>22969 AND ci.role_id<3 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>3069
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>624 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>8355 AND t.kind_id>1 AND t.production_year<1972
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=4331
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>407 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=15
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id>2440 AND t.kind_id<7 AND t.production_year>1994
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>8 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3376294 AND ci.role_id>1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2856402 AND ci.role_id=1 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=7 AND t.production_year<2008
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1126244
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3304581
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=2839778 AND t.kind_id>1 AND t.production_year<2004
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2721683 AND ci.role_id=1 AND t.kind_id=1 AND t.production_year=1995
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3488679
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>5
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<941 AND mk.keyword_id=11971
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=807
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>14916 AND t.kind_id<2
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>72945
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>1495112 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1140197
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1233649
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2912205 AND ci.role_id=1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=583627
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=18
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<26868 AND mi.info_type_id=16
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3798640
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=30633
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<33559 AND t.kind_id=7 AND t.production_year>1996
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<2623224 AND mc.company_id<71711
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id<100 AND mk.keyword_id>4095 AND t.kind_id=1 AND t.production_year>2012
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>645283 AND t.kind_id<7 AND t.production_year>2010
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id=1451 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.kind_id=1 AND t.production_year<1986
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id=74588 AND t.kind_id=2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3738753
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>419 AND t.kind_id<7 AND t.production_year<2009
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year=2008
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<3123128
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=6
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=49177
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=7 AND t.production_year>1986
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=1535
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.kind_id=1 AND t.production_year=2003
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=100 AND t.kind_id<7 AND t.production_year>1970
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>2 AND ci.person_id>1548190 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<13113
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id<3 AND t.production_year<2007
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id=3063603
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>74784 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<687 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<3 AND t.kind_id<7 AND t.production_year<2009
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<145
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=844
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>8478
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>899953
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1024956
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year<1966
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=4 AND t.production_year>2004
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2698508 AND ci.role_id=2
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id=8751 AND t.kind_id<4 AND t.production_year>1998
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id=229525 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>72200 AND mk.keyword_id<4952 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND mi.info_type_id<4
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND t.kind_id<4 AND t.production_year>2003
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND mc.company_id=19 AND t.kind_id=7 AND t.production_year=2013
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<131212
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3286498
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<52 AND mc.company_type_id<2 AND mk.keyword_id=16195 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1872364
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id<7 AND t.production_year=2010
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<47968 AND t.kind_id=7 AND t.production_year>2008
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<32337
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2239986 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>380620
SELECT COUNT(*) FROM title t WHERE t.kind_id=6 AND t.production_year<1998
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.kind_id=7 AND t.production_year=1991
SELECT COUNT(*) FROM title t WHERE t.kind_id=1 AND t.production_year=2000
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>15929
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>891693 AND ci.role_id=3
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=1 AND t.production_year>2010
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<93715 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=11146 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND t.kind_id>2 AND t.production_year=1949
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>521
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>74463
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<2 AND t.kind_id=1 AND t.production_year<1990
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=7 AND t.production_year>2002
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id=29530 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>910508
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1110377 AND t.kind_id=7 AND t.production_year=2005
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>17 AND mk.keyword_id>56
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<20493
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id>3 AND t.production_year<2007
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>160 AND mc.company_type_id>1 AND t.kind_id=3 AND t.production_year>2005
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<2174 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>101 AND mk.keyword_id<4678
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>14503
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=9779
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>11 AND mc.company_type_id>1 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id>10982 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year=2008
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>613
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=761037
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>8 AND t.kind_id<7 AND t.production_year>2011
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>14537
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>1815091 AND ci.role_id<10 AND mi_idx.info_type_id<100 AND t.kind_id=4 AND t.production_year<2011
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=1
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND t.kind_id=7 AND t.production_year>2012
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.kind_id<7 AND t.production_year=1975
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<3868 AND t.kind_id<7 AND t.production_year=1920
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND mi.info_type_id=7 AND t.kind_id>1 AND t.production_year<2005
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>829647 AND ci.role_id=4 AND mi_idx.info_type_id=100 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.kind_id<7 AND t.production_year=2005
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<2812673 AND ci.role_id<2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=815019
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>637724
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.kind_id<7 AND t.production_year=2011
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id<529723 AND t.kind_id<7 AND t.production_year=1999
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>117 AND t.kind_id<7 AND t.production_year=1970
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year>2009
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=3314023 AND ci.role_id>3 AND mk.keyword_id<103
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id=4 AND mi_idx.info_type_id<100
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2855174 AND ci.role_id=10 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id=3939 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1404759 AND ci.role_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>1680
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<928815 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id<7 AND t.production_year<2012
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<7121 AND mc.company_type_id=1
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<1447122 AND t.kind_id=7 AND t.production_year=2007
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year=2012
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<1075616 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=69570
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=6393
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>2226117
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2474223
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=13095
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND mk.keyword_id>3665 AND t.kind_id<7 AND t.production_year>1967
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3474965
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<733783 AND mk.keyword_id=2488 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1477842 AND ci.role_id<10 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id<480 AND mc.company_type_id=2 AND t.kind_id=1 AND t.production_year>2006
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id<1454 AND mc.company_type_id=1 AND ci.person_id>2751704
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>1 AND t.production_year>1959
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1709520 AND ci.role_id>4
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id<4 AND t.kind_id<2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1085601
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16 AND t.kind_id=7 AND t.production_year<2014
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>2241
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=7850
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<106018 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<13053 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>34216 AND mc.company_type_id>1 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id<21550 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2976496
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id>2284685
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=8581
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>101 AND t.kind_id<2
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>76379 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=1 AND t.production_year=1997
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<763 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=142225 AND ci.role_id=1 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id<15947 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1725766
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=2890669 AND ci.role_id>6
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id>641
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year>2011
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<24675
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3482175
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<804
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=21034
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=5 AND t.kind_id=7 AND t.production_year=2007
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2029272 AND ci.role_id>1
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id=1822304 AND ci.role_id=2
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id>98 AND t.kind_id<4 AND t.production_year>1958
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id<2553101
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1465663 AND ci.role_id<8
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id<6 AND t.production_year>2005
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3339531 AND ci.role_id>10
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<16264 AND t.kind_id=1 AND t.production_year=2013
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>258 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>112290 AND ci.person_id>2844446 AND ci.role_id>3 AND t.kind_id<7 AND t.production_year>0
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id>202769 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<3 AND t.production_year>2013
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id>16 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>8918 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=2 AND t.production_year>2011
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id=7 AND t.production_year>1975
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3721670 AND ci.role_id>6
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id=3 AND t.kind_id<7 AND t.production_year<2012
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>2799731 AND mk.keyword_id>16264 AND t.kind_id>1 AND t.production_year=2007
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id=4
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1832526 AND ci.role_id=2
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>4 AND mk.keyword_id=783 AND t.kind_id<7 AND t.production_year<2010
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>878 AND t.kind_id>1 AND t.production_year>2007
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=1 AND t.production_year>1952
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2459502 AND ci.role_id>2 AND t.kind_id=7
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year=2004
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=1133
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id<167753 AND t.kind_id=1 AND t.production_year<1985
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id=20891
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>734945 AND mk.keyword_id<4720 AND t.kind_id>1 AND t.production_year=2005
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id=5593
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<28064
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1286083
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=98 AND t.kind_id=1 AND t.production_year=2012
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<120401 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND ci.person_id<470578 AND ci.role_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<78886
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1513355
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=2667904
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id=73
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<2 AND t.production_year=1999
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND t.kind_id=7 AND t.production_year>1973
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<11156 AND t.kind_id=1 AND t.production_year>1991
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id<862
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>101 AND t.kind_id<3 AND t.production_year=1968
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<25793
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<394 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=4340
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND ci.person_id>1780025 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2371041
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>9868 AND mi_idx.info_type_id=99 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>1 AND t.production_year>1994
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=19 AND mc.company_type_id>1 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>2520473 AND ci.role_id>2 AND mi_idx.info_type_id<101 AND t.kind_id=3 AND t.production_year<1999
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=3010078
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>5603
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>154427 AND mc.company_type_id>1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1980119 AND ci.role_id>1 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<19952 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id>2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2860554 AND ci.role_id=5
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id=382
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id<429
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<55131
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND t.kind_id=2 AND t.production_year>1969
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1103506 AND ci.role_id<8 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<73 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<8420 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>24257
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<2 AND t.production_year<1983
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>19438
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<980247 AND t.kind_id=1
SELECT COUNT(*) FROM title t WHERE t.kind_id>2 AND t.production_year=2011
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id<423
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>1809 AND t.kind_id=7 AND t.production_year=2000
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>6014
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>3245846
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=85256 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id<15
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND mi.info_type_id=4 AND t.kind_id=3
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id=2196150
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2465476 AND ci.role_id<10
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id>4649 AND t.kind_id=7 AND t.production_year=2012
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=99 AND mk.keyword_id=282 AND t.kind_id=1 AND t.production_year>0
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3099474
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<6561
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=3 AND mk.keyword_id<9796 AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>14840
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id<1799
SELECT COUNT(*) FROM title t WHERE t.kind_id=2 AND t.production_year=2006
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=8679 AND t.kind_id=7 AND t.production_year<2008
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>3474 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=2030
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<872
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1847049
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<16807 AND t.kind_id=1 AND t.production_year=2005
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<159588 AND mk.keyword_id>7078
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id=51235
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id<16 AND mk.keyword_id>23686 AND t.kind_id=7 AND t.production_year=1953
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<3397
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2292389 AND ci.role_id<11
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<2440
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=10474 AND t.kind_id=7 AND t.production_year>2010
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>3549059 AND ci.role_id>3 AND t.kind_id>4 AND t.production_year>2008
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1595938
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.kind_id<7 AND t.production_year<1994
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id<3164799
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>2488 AND t.kind_id<7 AND t.production_year<1974
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<55553 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2963682
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mi_idx.info_type_id=101 AND t.kind_id>3 AND t.production_year=2011
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>99 AND mk.keyword_id=78489
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>2023 AND mc.company_type_id=2 AND ci.person_id=2846216
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=868 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=7 AND t.production_year<1994
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>3
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>2921814 AND ci.role_id>2 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1496384
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2820542 AND ci.role_id>1 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<1795196 AND t.kind_id<7 AND t.production_year>2007
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=198336 AND ci.role_id=4
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id=5 AND ci.person_id>814058 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1157476 AND ci.role_id>2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2488379 AND t.kind_id<3 AND t.production_year=1995
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=12533 AND t.kind_id=1 AND t.production_year<2004
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=292 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>99 AND t.kind_id>2
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id<7 AND t.production_year<2011
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id=566114 AND mi.info_type_id>7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=8 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=13543
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id>18
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id=3313581 AND mc.company_id>12786 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3680620 AND ci.role_id<3
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id=1527 AND mi_idx.info_type_id=99
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id=139832 AND mk.keyword_id>494
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id<7 AND t.production_year>2006
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1713649
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<239 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id>2054902
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id>15105
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2050560
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=160 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<5 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2011377 AND ci.role_id>2
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>599248 AND ci.role_id>10
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2688179
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>12821
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mk.keyword_id=3366
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<3353077 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>1 AND t.production_year<2008
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id>8064
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<4 AND t.production_year=2006
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=18219
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101 AND mk.keyword_id<7763 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<2347430 AND mi.info_type_id>8 AND t.kind_id>1
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<71983 AND mi_idx.info_type_id>99 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>1876
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=7 AND t.kind_id=7 AND t.production_year<1986
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<8678
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id=7105
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id<7640
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<13527
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=738
SELECT COUNT(*) FROM title t WHERE t.kind_id>4 AND t.production_year>1998
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>46128 AND t.kind_id>4
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id>2488 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3423829
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND mk.keyword_id>41526 AND t.kind_id=7 AND t.production_year>2011
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>71145 AND t.kind_id<4 AND t.production_year<2004
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<528784 AND ci.role_id<10
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND t.kind_id=4 AND t.production_year=2012
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>1 AND t.production_year=2011
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>2 AND t.production_year<2012
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id=99
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>41
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1074240 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id<86445
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id>1 AND t.production_year>1996
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<186
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi.info_type_id>7
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>6911
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id>1 AND t.production_year>2008
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2473150 AND ci.role_id=2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<16232
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id>36392 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1225403 AND ci.role_id<9 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<17646 AND mc.company_type_id<2 AND mk.keyword_id>17747
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>842
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=1429124 AND ci.role_id>3 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND ci.person_id<3382021 AND ci.role_id=4
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<11387 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id>329 AND ci.person_id>2123316
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<499269 AND ci.role_id>3 AND mk.keyword_id>29129
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>249 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1089766 AND ci.role_id<5
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>169996
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>46 AND mc.company_type_id>1
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id<2365533 AND ci.role_id>1 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>121 AND t.kind_id=7
SELECT COUNT(*) FROM title t WHERE t.kind_id=1 AND t.production_year>2007
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1717680 AND ci.role_id=9
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>745
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND t.kind_id<7 AND t.production_year<1990
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2333062
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id=2130 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND t.kind_id>4
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3848160 AND ci.role_id=2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<9003 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>797
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id>100 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<6254 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>81119
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year>1921
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>2803 AND mc.company_type_id=2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3022699 AND ci.role_id>2
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=196
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id>1 AND t.production_year>1971
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=917
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id>7349 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>427196 AND ci.role_id<4 AND t.kind_id>1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<3276274 AND mk.keyword_id<3020 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>8 AND mc.company_id>11137 AND t.kind_id<7 AND t.production_year>2007
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=8 AND mk.keyword_id<3243 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>2 AND t.kind_id=1 AND t.production_year>1979
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id<615439 AND ci.role_id>8 AND t.kind_id=1 AND t.production_year<2008
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id=1119308 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND t.kind_id>1
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year<1996
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1633666
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id<4384 AND t.kind_id<7 AND t.production_year<1961
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>347
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<1434 AND t.kind_id=7 AND t.production_year<2002
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2179643 AND t.kind_id<3 AND t.production_year<2005
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year=2011
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>100 AND mk.keyword_id<129120 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id>18 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=9055
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND t.kind_id=2
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>107 AND mc.company_id>6437 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<16 AND t.kind_id=7 AND t.production_year>2012
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id>2 AND t.production_year>2005
SELECT COUNT(*) FROM movie_companies mc,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mc.movie_id AND mc.company_id<58088
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=458088
SELECT COUNT(*) FROM title t WHERE t.kind_id=1 AND t.production_year<1999
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=1402 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<34 AND mc.company_type_id>1 AND mk.keyword_id<1382
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>2 AND mc.company_id<809 AND mc.company_type_id=2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1380001 AND ci.role_id<3
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>1141
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=16966
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year=1909
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>71001
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>911 AND mc.company_type_id=2 AND mk.keyword_id>643
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>143
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id=42
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id<7 AND t.kind_id>1 AND t.production_year<2011
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<3496811 AND ci.role_id<3 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id>15807 AND mc.company_type_id>1
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND mk.keyword_id=27542
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<18349 AND mc.company_type_id<2 AND t.kind_id<4 AND t.production_year<1993
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1336135
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>108 AND mi_idx.info_type_id>99 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mk.keyword_id<335 AND t.kind_id>2
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=91411 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id>3796 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2800409
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id=7 AND t.production_year<2011
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND t.kind_id>1 AND t.production_year<2011
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3756564
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<49 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<8 AND t.kind_id=4 AND t.production_year>2009
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.kind_id<3 AND t.production_year=2003
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=5720 AND t.kind_id=7 AND t.production_year<2006
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>599378 AND ci.role_id=10
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>1166025 AND ci.role_id=1
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<2908541 AND ci.role_id>10
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id<101 AND t.kind_id>3 AND t.production_year>2010
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND mc.company_id<13259 AND mc.company_type_id<2 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.kind_id<7 AND t.production_year>2011
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<89695
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=17 AND mk.keyword_id>641 AND t.kind_id<2 AND t.production_year=2011
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>222351 AND t.kind_id<2
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>1 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>5309
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>25696
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1567811
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<895 AND mc.company_type_id>1
SELECT COUNT(*) FROM title t WHERE t.kind_id=3 AND t.production_year=1995
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3080099
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id>169533 AND mc.company_type_id=2 AND t.kind_id<7 AND t.production_year<1970
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<4668
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>521 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year=2007
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>2908
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.kind_id<3
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>313993 AND ci.role_id=8
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<651520 AND mk.keyword_id>14517
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year=2010
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id>71101 AND t.kind_id=7 AND t.production_year=1970
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<653680 AND t.kind_id>3
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>31051
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<14002 AND mc.company_type_id>1 AND t.kind_id<7 AND t.production_year<2010
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>4601
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>226651
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<67383
SELECT COUNT(*) FROM title t WHERE t.kind_id>1 AND t.production_year=2008
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=7986
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=11137
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=17641
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<1138
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1315983
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>3219
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id=76379
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=154010
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<29029
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=5898 AND mc.company_type_id>1 AND t.kind_id<7 AND t.production_year<2012
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=67
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND mc.company_id>201 AND mc.company_type_id>1 AND t.kind_id<7 AND t.production_year<2011
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2696055 AND t.kind_id=1
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<7 AND t.production_year>1972
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=2800491
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101 AND mk.keyword_id=58
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id<2101306 AND ci.role_id<4
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=101 AND mi.info_type_id<107 AND t.kind_id>3 AND t.production_year>1978
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>93223 AND t.kind_id<7 AND t.production_year>1977
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>650188 AND t.kind_id<7 AND t.production_year>1935
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1348953 AND ci.role_id>4
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id>2497 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>916
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id>1600671 AND ci.role_id=2 AND t.kind_id=1 AND t.production_year=1915
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<1145008
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2098230
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>1 AND t.production_year<2003
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=99 AND mk.keyword_id>835 AND t.kind_id=1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1837394 AND ci.role_id>4
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND t.kind_id<7 AND t.production_year=2007
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1785503 AND ci.role_id>8
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id=4 AND t.kind_id>2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>704841
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<95397 AND ci.role_id>2 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id>3 AND mc.company_id<483 AND mc.company_type_id=2 AND t.kind_id<7 AND t.production_year<2010
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<16452
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<1684
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id<11272 AND mc.company_type_id=1 AND mi_idx.info_type_id<100
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id>54750 AND t.kind_id<7 AND t.production_year>1912
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mk.keyword_id<78044 AND t.kind_id<7 AND t.production_year=2003
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>41090 AND ci.role_id<4 AND mi_idx.info_type_id=100 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=7 AND t.production_year=1968
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>76445
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=4037 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id<4 AND mi_idx.info_type_id<101 AND t.kind_id>4 AND t.production_year=1989
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=964726
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<3487034 AND ci.role_id>2 AND t.kind_id<7 AND t.production_year<1996
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>2 AND mk.keyword_id<112966 AND t.kind_id<7 AND t.production_year>1999
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>472102 AND mk.keyword_id>335
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>8566 AND mc.company_type_id<2
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id>879397 AND mk.keyword_id>4037 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>6 AND mk.keyword_id=36 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<9557
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<6845
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id=3 AND t.production_year<2008
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>475810 AND ci.role_id>2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=101
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id<2341611 AND ci.role_id=8 AND t.kind_id<7 AND t.production_year=1991
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>9757 AND mc.company_type_id=2 AND t.kind_id=2 AND t.production_year=2012
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id<7 AND t.production_year<2013
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>11456 AND mc.company_type_id=2 AND mk.keyword_id>8605 AND t.kind_id>1 AND t.production_year=1970
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id>2 AND t.kind_id=7 AND t.production_year<1994
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2956927 AND ci.role_id=3
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>567945 AND mi_idx.info_type_id=101 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id>100 AND mk.keyword_id>23091
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<3660 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=14490
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id<1157189 AND ci.role_id<2 AND t.kind_id=7
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year<1913
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=213
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<118647
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>2267878 AND ci.role_id<4 AND mk.keyword_id<5878
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id=1268732
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=139
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id<5117
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<101 AND mc.company_id<7108 AND mc.company_type_id=2
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3550163
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>1925916 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1894472 AND ci.role_id>3
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND t.kind_id=7 AND t.production_year<1989
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>90276 AND t.kind_id>1 AND t.production_year=1959
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=1 AND t.production_year>1935
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1185139
SELECT COUNT(*) FROM title t WHERE t.kind_id>4 AND t.production_year=2012
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year<1963
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>3 AND t.kind_id>1 AND t.production_year=1982
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<47260
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>6920 AND mc.company_type_id<2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>101
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<11214 AND mi.info_type_id=16 AND t.kind_id=7 AND t.production_year>2012
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>72478
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1494142 AND ci.role_id=10
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>917 AND mc.company_type_id=1 AND t.kind_id>2 AND t.production_year<2011
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3496472
SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_id<85
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND t.kind_id<7 AND t.production_year<2003
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<3053898
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=71001
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<71922
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<76674 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id<9010
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<43954
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year=1989
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mk.keyword_id<2281 AND t.kind_id=4 AND t.production_year=2006
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<1877 AND mc.company_type_id<2 AND mk.keyword_id<679 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id<7 AND t.production_year=1967
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>225530
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id=106
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3769285
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id=3665 AND t.kind_id=4
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id<67 AND mk.keyword_id=1138
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND mi.info_type_id=8 AND t.kind_id<7
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=216
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<5835 AND mc.company_type_id>1 AND mi.info_type_id>16 AND t.kind_id=7 AND t.production_year>1965
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<6623
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id>98 AND t.kind_id=7
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<139207
SELECT COUNT(*) FROM cast_info ci,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=ci.movie_id AND ci.person_id=778377 AND t.kind_id>4
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=198249
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=80834
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<3172
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3414197
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3164119
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id=7 AND mk.keyword_id>42252 AND t.kind_id=2 AND t.production_year<2009
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=39637
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<4691 AND t.kind_id>1 AND t.production_year>2001
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=536
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=8 AND mc.company_id>19458 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>8969 AND mi.info_type_id=2 AND t.kind_id<7 AND t.production_year<2003
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3449303
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi.info_type_id>104 AND mi_idx.info_type_id>99 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mk.keyword_id<2879
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND ci.person_id=1327388
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mc.movie_id AND mc.company_id>474
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>769
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>683
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2536314
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>2763 AND mc.company_type_id<2 AND mi.info_type_id>16 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id>1 AND t.production_year=1956
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<721717 AND t.kind_id=3
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year=2013
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND t.kind_id<7 AND t.production_year=1996
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>3783139 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1150485
SELECT COUNT(*) FROM title t WHERE t.kind_id=3 AND t.production_year=2008
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<27 AND mc.company_type_id=2 AND t.kind_id<7
SELECT COUNT(*) FROM title t WHERE t.kind_id=4 AND t.production_year>2004
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id=19 AND t.kind_id>1 AND t.production_year=2007
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>76163 AND mk.keyword_id=17222 AND t.kind_id<2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id<101
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND ci.person_id>1155982 AND ci.role_id=2 AND mi_idx.info_type_id=101
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND mk.keyword_id>37136
SELECT COUNT(*) FROM movie_info_idx mi_idx,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=6 AND t.production_year<1968
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND t.kind_id=3 AND t.production_year>2005
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<18214
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id<7 AND t.production_year>2011
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<6929 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1741807 AND ci.role_id>8
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>35692
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>11504 AND mc.company_type_id=1 AND t.kind_id=1
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year=2009
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year=1994
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.kind_id>6
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id>2930 AND mk.keyword_id<5760
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<13629 AND mc.company_type_id=1 AND t.kind_id=1 AND t.production_year<1999
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>3205684
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id=1 AND t.production_year<2011
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<20656 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>15470
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year>2003
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id<6316 AND t.kind_id=7 AND t.production_year>2005
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<3026
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2389469
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2476453 AND ci.role_id=10
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<3095275
SELECT COUNT(*) FROM title t WHERE t.kind_id=7 AND t.production_year=1973
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>1732
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=85474
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<47694
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>20863 AND t.kind_id>1 AND t.production_year=2011
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id=1451
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>1 AND t.production_year>1961
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id<1839357 AND t.kind_id=7
SELECT COUNT(*) FROM title t WHERE t.kind_id=1 AND t.production_year>2004
SELECT COUNT(*) FROM cast_info ci,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=ci.movie_id AND ci.person_id<2268493 AND ci.role_id<2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year=2000
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=3085
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<77923
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<2016344 AND t.kind_id<7 AND t.production_year<2005
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=30740
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id=11928 AND mc.company_type_id>1 AND mi.info_type_id>1 AND t.kind_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>7776
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id>1284
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id>1848517 AND t.kind_id=3 AND t.production_year=2004
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<16 AND t.kind_id>3 AND t.production_year<1954
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>1 AND t.production_year>1953
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=1039285
SELECT COUNT(*) FROM movie_info mi,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mi.movie_id AND mi_idx.info_type_id<100
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND mc.company_id<166641 AND t.kind_id>1 AND t.production_year>2008
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id>47874 AND mc.company_type_id=2 AND t.kind_id<2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mc.company_id=323
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=7608 AND t.kind_id<7 AND t.production_year<2012
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=1113
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<157183 AND mc.company_type_id>1
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id>741
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>12540 AND mc.company_type_id>1
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<2480886
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year>2005
SELECT COUNT(*) FROM movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.kind_id<7 AND t.production_year=2006
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>683645
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>479
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>16264
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=667
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=1935
SELECT COUNT(*) FROM title t WHERE t.kind_id<6 AND t.production_year<2007
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=3 AND t.production_year=2003
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<1897706 AND ci.role_id=1 AND t.kind_id<7 AND t.production_year<1995
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>4841
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id>100 AND mk.keyword_id<2358
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id>3 AND t.production_year=1961
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND ci.person_id<1201181 AND ci.role_id=2 AND t.kind_id=1
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id>28339 AND mi.info_type_id=16
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>1082368
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mi.info_type_id>16 AND t.kind_id<7 AND t.production_year>0
SELECT COUNT(*) FROM movie_companies mc,movie_keyword mk,title t WHERE t.id=mc.movie_id AND t.id=mk.movie_id AND t.kind_id<7 AND t.production_year>2012
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id<100 AND mk.keyword_id=622
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<15 AND t.kind_id=1
SELECT COUNT(*) FROM title t WHERE t.kind_id<7 AND t.production_year<1972
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND t.kind_id>4 AND t.production_year=2012
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id=16 AND mc.company_id=129 AND t.kind_id>1 AND t.production_year<1996
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id<1451 AND t.kind_id=1 AND t.production_year=2007
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=624
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>45346
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND t.kind_id=7 AND t.production_year=2004
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>6230 AND t.kind_id=7 AND t.production_year=2012
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<17 AND t.kind_id<7
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>92
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id<34119 AND t.kind_id<7 AND t.production_year=1968
SELECT COUNT(*) FROM movie_info mi,cast_info ci,title t WHERE t.id=ci.movie_id AND t.id=mi.movie_id AND mi.info_type_id>3 AND t.kind_id>3
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND mc.company_id>5197 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id<46332 AND t.kind_id<7 AND t.production_year<2002
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id<100 AND mi.info_type_id>16 AND t.kind_id<7
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id=101 AND t.kind_id=3
SELECT COUNT(*) FROM movie_companies mc,movie_info mi,title t WHERE t.id=mi.movie_id AND t.id=mc.movie_id AND mc.company_id<19 AND mc.company_type_id<2 AND mi.info_type_id<4 AND t.kind_id<7 AND t.production_year=1988
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.kind_id=2
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>214289
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id=77506
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=5547
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id=7 AND t.kind_id>1 AND t.production_year=2008
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id>13135 AND t.kind_id=7
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=1383
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id<205213 AND mc.company_type_id=1
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id=348
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1757930 AND ci.role_id=2
SELECT COUNT(*) FROM movie_companies mc,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=mc.movie_id AND mc.company_id=13656 AND t.kind_id<7
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mi.movie_id AND t.id=mk.movie_id AND mi.info_type_id>16 AND t.kind_id=1 AND t.production_year=1978
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<2032112 AND t.kind_id>1
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mi_idx.info_type_id=100 AND mk.keyword_id>245
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>34728
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>448 AND mc.company_type_id=2
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>3612
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id<1236752
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<2989
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id>1382
SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND mi.info_type_id<9 AND t.kind_id>1 AND t.production_year=2010
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2598274
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=3849783
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mc.company_id=6 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND t.kind_id<7 AND t.production_year<1994
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mi_idx.movie_id AND t.id=mk.movie_id AND mk.keyword_id=14343 AND t.kind_id<2
SELECT COUNT(*) FROM movie_info_idx mi_idx,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=100 AND mk.keyword_id=57698
SELECT COUNT(*) FROM movie_companies mc,title t WHERE t.id=mc.movie_id AND t.kind_id>4 AND t.production_year>1974
SELECT COUNT(*) FROM movie_info mi,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=mi.movie_id AND mi.info_type_id=7 AND t.kind_id=7 AND t.production_year=2003
SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_id>2623
SELECT COUNT(*) FROM cast_info ci,movie_info_idx mi_idx,title t WHERE t.id=mi_idx.movie_id AND t.id=ci.movie_id AND mi_idx.info_type_id<100 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>2177827
SELECT COUNT(*) FROM cast_info ci,title t WHERE t.id=ci.movie_id AND ci.person_id>2742752 AND ci.role_id>2 AND t.kind_id=7
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=ci.movie_id AND t.id=mk.movie_id AND ci.person_id<1717758 AND ci.role_id>6 AND mk.keyword_id<348 AND t.kind_id<4
SELECT COUNT(*) FROM movie_info mi,movie_companies mc,title t WHERE t.id=mc.movie_id AND t.id=mi.movie_id AND mi.info_type_id<16 AND mc.company_id<46
SELECT COUNT(*) FROM cast_info ci,movie_keyword mk,title t WHERE t.id=mk.movie_id AND t.id=ci.movie_id AND ci.person_id<3027337 AND ci.role_id<2 AND t.kind_id=7 AND t.production_year>2013
SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id=2219637
SELECT COUNT(*) FROM movie_keyword mk,title t WHERE t.id=mk.movie_id AND mk.keyword_id=1291
SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_id<2893
