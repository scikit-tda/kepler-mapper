This page is broken


Parameters
----------

Initialize
~~~~~~~~~~

.. code:: python

    mapper = km.KeplerMapper(verbose=1)

+-------------+-----------------------------------------------+
| Parameter   | Description                                   |
+=============+===============================================+
| verbose     | Int. Verbosity of the mapper. *Default = 0*   |
+-------------+-----------------------------------------------+

Fitting and transforming
~~~~~~~~~~~~~~~~~~~~~~~~

Input the data set. Specify a projection/lens type. Output the projected
data/lens.

.. code:: python

    projected_data = mapper.fit_transform(data, projection="sum",
                                          scaler=km.preprocessing.MinMaxScaler() )

+------+------+
| Para | Desc |
| mete | ript |
| r    | ion  |
+======+======+
| data | Nump |
|      | y    |
|      | Arra |
|      | y.   |
|      | The  |
|      | data |
|      | to   |
|      | fit  |
|      | a    |
|      | proj |
|      | ecti |
|      | on/l |
|      | ens  |
|      | to.  |
|      | *Req |
|      | uire |
|      | d*   |
+------+------+
| proj | Any  |
| ecti | of:  |
| on   | list |
|      | with |
|      | dime |
|      | nsio |
|      | n    |
|      | indi |
|      | ces. |
|      | Scik |
|      | it-l |
|      | earn |
|      | API  |
|      | comp |
|      | atib |
|      | le   |
|      | mani |
|      | fold |
|      | lear |
|      | ner  |
|      | or   |
|      | dime |
|      | nsio |
|      | nali |
|      | ty   |
|      | redu |
|      | cer. |
|      | A    |
|      | stri |
|      | ng   |
|      | from |
|      | ["su |
|      | m"," |
|      | mean |
|      | ","m |
|      | edia |
|      | n"," |
|      | max" |
|      | ,"mi |
|      | n"," |
|      | std" |
|      | ,"di |
|      | st\_ |
|      | mean |
|      | ","l |
|      | 2nor |
|      | m"," |
|      | knn\ |
|      | _dis |
|      | tanc |
|      | e\_n |
|      | "].  |
|      | If   |
|      | usin |
|      | g    |
|      | ``kn |
|      | n_di |
|      | stan |
|      | ce_n |
|      | ``   |
|      | writ |
|      | e    |
|      | the  |
|      | numb |
|      | er   |
|      | of   |
|      | desi |
|      | red  |
|      | neig |
|      | hbor |
|      | s    |
|      | in   |
|      | plac |
|      | e    |
|      | of   |
|      | ``n` |
|      | `:   |
|      | ``kn |
|      | n_di |
|      | stan |
|      | ce_5 |
|      | ``   |
|      | for  |
|      | summ |
|      | ed   |
|      | dist |
|      | ance |
|      | s    |
|      | to 5 |
|      | near |
|      | est  |
|      | neig |
|      | hbor |
|      | s.   |
|      | *Def |
|      | ault |
|      | =    |
|      | "sum |
|      | "*.  |
+------+------+
| scal | Scik |
| er   | it-L |
|      | earn |
|      | API  |
|      | comp |
|      | atib |
|      | le   |
|      | scal |
|      | er.  |
|      | Scal |
|      | er   |
|      | of   |
|      | the  |
|      | data |
|      | appl |
|      | ied  |
|      | befo |
|      | re   |
|      | mapp |
|      | ing. |
|      | Use  |
|      | ``No |
|      | ne`` |
|      | for  |
|      | no   |
|      | scal |
|      | ing. |
|      | *Def |
|      | ault |
|      | =    |
|      | prep |
|      | roce |
|      | ssin |
|      | g.Mi |
|      | nMax |
|      | Scal |
|      | er() |
|      | *    |
+------+------+
| dist | ``Fa |
| ance | lse` |
| \_ma | `    |
| trix | or   |
|      | any  |
|      | of:  |
|      | ["br |
|      | aycu |
|      | rtis |
|      | ",   |
|      | "can |
|      | berr |
|      | a",  |
|      | "che |
|      | bysh |
|      | ev", |
|      | "cit |
|      | yblo |
|      | ck", |
|      | "cor |
|      | rela |
|      | tion |
|      | ",   |
|      | "cos |
|      | ine" |
|      | ,    |
|      | "dic |
|      | e",  |
|      | "euc |
|      | lide |
|      | an", |
|      | "ham |
|      | ming |
|      | ",   |
|      | "jac |
|      | card |
|      | ",   |
|      | "kul |
|      | sins |
|      | ki", |
|      | "mah |
|      | alan |
|      | obis |
|      | ",   |
|      | "mat |
|      | chin |
|      | g",  |
|      | "min |
|      | kows |
|      | ki", |
|      | "rog |
|      | erst |
|      | anim |
|      | oto" |
|      | ,    |
|      | "rus |
|      | sell |
|      | rao" |
|      | ,    |
|      | "seu |
|      | clid |
|      | ean" |
|      | ,    |
|      | "sok |
|      | almi |
|      | chen |
|      | er", |
|      | "sok |
|      | alsn |
|      | eath |
|      | ",   |
|      | "sqe |
|      | ucli |
|      | dean |
|      | ",   |
|      | "yul |
|      | e"]. |
|      | If   |
|      | ``Fa |
|      | lse` |
|      | `    |
|      | do   |
|      | noth |
|      | ing, |
|      | else |
|      | crea |
|      | te   |
|      | a    |
|      | squa |
|      | red  |
|      | dist |
|      | ance |
|      | matr |
|      | ix   |
|      | with |
|      | the  |
|      | chos |
|      | en   |
|      | metr |
|      | ic,  |
|      | befo |
|      | re   |
|      | appl |
|      | ying |
|      | the  |
|      | proj |
|      | ecti |
|      | on.  |
+------+------+

Mapping
~~~~~~~

.. code:: python

    topological_network = mapper.map(projected_X, inverse_X=None,
                                     clusterer=cluster.DBSCAN(eps=0.5,min_samples=3),
                                     nr_cubes=10, overlap_perc=0.1)

    print(topological_network["nodes"])
    print(topological_network["links"])
    print(topological_network["meta"])

+------+------+
| Parameter | Description |
+======+======+
| projected\_X | Numpy array. Output from fit\_transform. *Required*|
+------+------+
| inve | Nump |
| rse\ | y    |
| _X   | arra |
|      | y    |
|      | or   |
|      | ``No |
|      | ne`` |
|      | .    |
|      | When |
|      | ``No |
|      | ne`` |
|      | ,    |
|      | clus |
|      | ter  |
|      | on   |
|      | the  |
|      | proj |
|      | ecti |
|      | on,  |
|      | else |
|      | clus |
|      | ter  |
|      | on   |
|      | the  |
|      | orig |
|      | inal |
|      | data |
|      | (inv |
|      | erse |
|      | imag |
|      | e).  |
+------+------+
| clus | Scik |
| tere | it-L |
| r    | earn |
|      | API  |
|      | comp |
|      | atib |
|      | le   |
|      | clus |
|      | teri |
|      | ng   |
|      | algo |
|      | rith |
|      | m.   |
|      | The  |
|      | clus |
|      | teri |
|      | ng   |
|      | algo |
|      | rith |
|      | m    |
|      | to   |
|      | use  |
|      | for  |
|      | mapp |
|      | ing. |
|      | *Def |
|      | ault |
|      | =    |
|      | clus |
|      | ter. |
|      | DBSC |
|      | AN(e |
|      | ps=0 |
|      | .5,m |
|      | in\_ |
|      | samp |
|      | les= |
|      | 3)*  |
+------+------+
| nr\_ | Int. |
| cube | The  |
| s    | numb |
|      | er   |
|      | of   |
|      | cube |
|      | s/in |
|      | terv |
|      | als  |
|      | to   |
|      | crea |
|      | te.  |
|      | *Def |
|      | ault |
|      | =    |
|      | 10*  |
+------+------+
| over | Floa |
| lap\ | t.   |
| _per | How  |
| c    | much |
|      | the  |
|      | cube |
|      | s/in |
|      | terv |
|      | als  |
|      | over |
|      | lap  |
|      | (rel |
|      | evan |
|      | t    |
|      | for  |
|      | crea |
|      | ting |
|      | the  |
|      | edge |
|      | s).  |
|      | *Def |
|      | ault |
|      | =    |
|      | 0.1* |
+------+------+

Visualizing
~~~~~~~~~~~

.. code:: python

    mapper.visualize(topological_network,
                     path_html="mapper_visualization_output.html")

+------+------+
| Para | Desc |
| mete | ript |
| r    | ion  |
+======+======+
| topo | Dict |
| logi | .    |
| cal\ | The  |
| _net | ``to |
| work | polo |
|      | gica |
|      | l_ne |
|      | twor |
|      | k``- |
|      | dict |
|      | iona |
|      | ry   |
|      | with |
|      | node |
|      | s,   |
|      | edge |
|      | s    |
|      | and  |
|      | meta |
|      | -inf |
|      | orma |
|      | tion |
|      | .    |
|      | *Req |
|      | uire |
|      | d*   |
+------+------+
| path | File |
| \_ht | path |
| ml   | .    |
|      | Path |
|      | wher |
|      | e    |
|      | to   |
|      | outp |
|      | ut   |
|      | the  |
|      | .htm |
|      | l    |
|      | file |
|      | *Def |
|      | ault |
|      | =    |
|      | mapp |
|      | er\_ |
|      | visu |
|      | aliz |
|      | atio |
|      | n\_o |
|      | utpu |
|      | t.ht |
|      | ml*  |
+------+------+
| titl | Stri |
| e    | ng.  |
|      | Docu |
|      | ment |
|      | titl |
|      | e    |
|      | for  |
|      | use  |
|      | in   |
|      | the  |
|      | outp |
|      | utte |
|      | d    |
|      | .htm |
|      | l.   |
|      | *Def |
|      | ault |
|      | =    |
|      | "My  |
|      | Data |
|      | "*   |
+------+------+
| grap | Int. |
| h\_l | Glob |
| ink\ | al   |
| _dis | leng |
| tanc | th   |
| e    | of   |
|      | link |
|      | s    |
|      | betw |
|      | een  |
|      | node |
|      | s.   |
|      | Use  |
|      | less |
|      | for  |
|      | larg |
|      | er   |
|      | grap |
|      | hs.  |
|      | *Def |
|      | ault |
|      | =    |
|      | 30*  |
+------+------+
| grap | Int. |
| h\_c | The  |
| harg | char |
| e    | ge   |
|      | betw |
|      | een  |
|      | node |
|      | s.   |
|      | Use  |
|      | less |
|      | nega |
|      | tive |
|      | char |
|      | ge   |
|      | for  |
|      | larg |
|      | er   |
|      | grap |
|      | hs.  |
|      | *Def |
|      | ault |
|      | =    |
|      | -120 |
|      | *    |
+------+------+
| grap | Floa |
| h\_g | t.   |
| ravi | A    |
| ty   | weak |
|      | geom |
|      | etri |
|      | c    |
|      | cons |
|      | trai |
|      | nt   |
|      | simi |
|      | lar  |
|      | to a |
|      | virt |
|      | ual  |
|      | spri |
|      | ng   |
|      | conn |
|      | ecti |
|      | ng   |
|      | each |
|      | node |
|      | to   |
|      | the  |
|      | cent |
|      | er   |
|      | of   |
|      | the  |
|      | layo |
|      | ut's |
|      | size |
|      | .    |
|      | Don' |
|      | t    |
|      | you  |
|      | set  |
|      | to   |
|      | nega |
|      | tive |
|      | or   |
|      | it's |
|      | turt |
|      | les  |
|      | all  |
|      | the  |
|      | way  |
|      | up.  |
|      | *Def |
|      | ault |
|      | =    |
|      | 0.1* |
+------+------+
| cust | NumP |
| om\_ | y    |
| tool | Arra |
| tips | y.   |
|      | Crea |
|      | te   |
|      | cust |
|      | om   |
|      | tool |
|      | tips |
|      | for  |
|      | all  |
|      | the  |
|      | node |
|      | memb |
|      | ers. |
|      | You  |
|      | coul |
|      | d    |
|      | use  |
|      | the  |
|      | targ |
|      | et   |
|      | labe |
|      | ls   |
|      | ``y` |
|      | `    |
|      | for  |
|      | this |
|      | .    |
|      | Use  |
|      | ``No |
|      | ne`` |
|      | for  |
|      | stan |
|      | dard |
|      | tool |
|      | tips |
|      | .    |
|      | *Def |
|      | ault |
|      | =    |
|      | None |
|      | *.   |
+------+------+
| show | Bool |
| \_ti | .    |
| tle  | Whet |
|      | her  |
|      | to   |
|      | show |
|      | the  |
|      | titl |
|      | e.   |
|      | *Def |
|      | ault |
|      | =    |
|      | True |
|      | *    |
+------+------+
| show | Bool |
| \_me | .    |
| ta   | Whet |
|      | her  |
|      | to   |
|      | show |
|      | meta |
|      | info |
|      | rmat |
|      | ion, |
|      | like |
|      | the  |
|      | over |
|      | lap  |
|      | perc |
|      | enta |
|      | ge   |
|      | and  |
|      | the  |
|      | clus |
|      | tere |
|      | r    |
|      | used |
|      | .    |
|      | *Def |
|      | ault |
|      | =    |
|      | True |
|      | *    |
+------+------+
| show | Bool |
| \_to | .    |
| olti | Whet |
| ps   | her  |
|      | to   |
|      | show |
|      | the  |
|      | tool |
|      | tips |
|      | on   |
|      | hove |
|      | r.   |
|      | *Def |
|      | ault |
|      | =    |
|      | True |
|      | *    |
+------+------+
| widt | Int. |
| h\_h | Size |
| tml  | in   |
|      | pixe |
|      | ls   |
|      | of   |
|      | the  |
|      | grap |
|      | h    |
|      | canv |
|      | as   |
|      | widt |
|      | h.   |
|      | *Def |
|      | ault |
|      | = 0  |
|      | (ful |
|      | l    |
|      | scre |
|      | en   |
|      | widt |
|      | h)*  |
+------+------+
| heig | Int. |
| ht\_ | Size |
| html | in   |
|      | pixe |
|      | ls   |
|      | of   |
|      | the  |
|      | grap |
|      | h    |
|      | canv |
|      | as   |
|      | heig |
|      | ht.  |
|      | *Def |
|      | ault |
|      | = 0  |
|      | (ful |
|      | l    |
|      | scre |
|      | en   |
|      | heig |
|      | ht)* |
+------+------+
