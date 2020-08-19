.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_generated_gallery_plot_digits.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_generated_gallery_plot_digits.py:


Digits Dataset
================

This digits example shows two ways of customizing the tooltips options in the HTML visualization. It generates the visualization with tooltips set as the y-label, or number of the image. The second generated result uses the actual image in the tooltips. 

`Visualization with y-label tooltip <../../_static/digits_ylabel_tooltips.html>`_

`Visualization with custom tooltips <../../_static/digits_custom_tooltips.html>`_


.. image:: /generated/gallery/images/sphx_glr_plot_digits_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    KeplerMapper(verbose=2)
    ..Composing projection pipeline of length 1:
            Projections: TSNE(angle=0.5, early_exaggeration=12.0, init='random', learning_rate=200.0,
       method='barnes_hut', metric='euclidean', min_grad_norm=1e-07,
       n_components=2, n_iter=1000, n_iter_without_progress=300,
       perplexity=30.0, random_state=None, verbose=0)
            Distance matrices: False
            Scalers: MinMaxScaler(copy=True, feature_range=(0, 1))
    ..Projecting on data shaped (1797, 64)

    ..Projecting data using: 
            TSNE(angle=0.5, early_exaggeration=12.0, init='random', learning_rate=200.0,
       method='barnes_hut', metric='euclidean', min_grad_norm=1e-07,
       n_components=2, n_iter=1000, n_iter_without_progress=300,
       perplexity=30.0, random_state=None, verbose=2)

    [t-SNE] Computing 91 nearest neighbors...
    [t-SNE] Indexed 1797 samples in 0.002s...
    [t-SNE] Computed neighbors for 1797 samples in 0.311s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 1797
    [t-SNE] Computed conditional probabilities for sample 1797 / 1797
    [t-SNE] Mean sigma: 8.121136
    [t-SNE] Computed conditional probabilities in 0.089s
    [t-SNE] Iteration 50: error = 75.5607758, gradient norm = 0.1324439 (50 iterations in 0.987s)
    [t-SNE] Iteration 100: error = 64.0872498, gradient norm = 0.0531824 (50 iterations in 0.569s)
    [t-SNE] Iteration 150: error = 62.6140709, gradient norm = 0.0351771 (50 iterations in 0.554s)
    [t-SNE] Iteration 200: error = 62.1142731, gradient norm = 0.0276543 (50 iterations in 0.543s)
    [t-SNE] Iteration 250: error = 61.8839035, gradient norm = 0.0306578 (50 iterations in 0.542s)
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 61.883904
    [t-SNE] Iteration 300: error = 1.0669582, gradient norm = 0.0009094 (50 iterations in 0.526s)
    [t-SNE] Iteration 350: error = 0.8864648, gradient norm = 0.0004199 (50 iterations in 0.513s)
    [t-SNE] Iteration 400: error = 0.8296416, gradient norm = 0.0002803 (50 iterations in 0.509s)
    [t-SNE] Iteration 450: error = 0.8034588, gradient norm = 0.0002075 (50 iterations in 0.510s)
    [t-SNE] Iteration 500: error = 0.7891181, gradient norm = 0.0001700 (50 iterations in 0.511s)
    [t-SNE] Iteration 550: error = 0.7798692, gradient norm = 0.0001618 (50 iterations in 0.513s)
    [t-SNE] Iteration 600: error = 0.7733858, gradient norm = 0.0001280 (50 iterations in 0.513s)
    [t-SNE] Iteration 650: error = 0.7688195, gradient norm = 0.0001267 (50 iterations in 0.505s)
    [t-SNE] Iteration 700: error = 0.7650961, gradient norm = 0.0001246 (50 iterations in 0.502s)
    [t-SNE] Iteration 750: error = 0.7622283, gradient norm = 0.0001175 (50 iterations in 0.500s)
    [t-SNE] Iteration 800: error = 0.7591587, gradient norm = 0.0001101 (50 iterations in 0.498s)
    [t-SNE] Iteration 850: error = 0.7565409, gradient norm = 0.0001135 (50 iterations in 0.497s)
    [t-SNE] Iteration 900: error = 0.7549771, gradient norm = 0.0001124 (50 iterations in 0.492s)
    [t-SNE] Iteration 950: error = 0.7535065, gradient norm = 0.0000926 (50 iterations in 0.484s)
    [t-SNE] Iteration 1000: error = 0.7522733, gradient norm = 0.0000857 (50 iterations in 0.485s)
    [t-SNE] KL divergence after 1000 iterations: 0.752273

    ..Scaling with: MinMaxScaler(copy=True, feature_range=(0, 1))

    Mapping on data shaped (1797, 2) using lens shaped (1797, 2)

    Minimal points in hypercube before clustering: 15
    Creating 1225 hypercubes.
    Cube_0 is empty.

    Cube_1 is empty.

    Cube_2 is empty.

    Cube_3 is empty.

       > Found 1 clusters in hypercube 4.
       > Found 1 clusters in hypercube 5.
    Cube_6 is empty.

    Cube_7 is empty.

    Cube_8 is empty.

       > Found 1 clusters in hypercube 9.
    Cube_10 is empty.

    Cube_11 is empty.

    Cube_12 is empty.

       > Found 1 clusters in hypercube 13.
       > Found 1 clusters in hypercube 14.
       > Found 1 clusters in hypercube 15.
       > Found 1 clusters in hypercube 16.
       > Found 1 clusters in hypercube 17.
    Cube_18 is empty.

    Cube_19 is empty.

       > Found 1 clusters in hypercube 20.
       > Found 1 clusters in hypercube 21.
       > Found 1 clusters in hypercube 22.
    Cube_23 is empty.

    Cube_24 is empty.

       > Found 1 clusters in hypercube 25.
       > Found 1 clusters in hypercube 26.
       > Found 1 clusters in hypercube 27.
       > Found 1 clusters in hypercube 28.
       > Found 1 clusters in hypercube 29.
    Cube_30 is empty.

    Cube_31 is empty.

       > Found 1 clusters in hypercube 32.
       > Found 1 clusters in hypercube 33.
       > Found 1 clusters in hypercube 34.
       > Found 1 clusters in hypercube 35.
    Cube_36 is empty.

    Cube_37 is empty.

    Cube_38 is empty.

       > Found 1 clusters in hypercube 39.
       > Found 1 clusters in hypercube 40.
       > Found 1 clusters in hypercube 41.
       > Found 1 clusters in hypercube 42.
    Cube_43 is empty.

    Cube_44 is empty.

       > Found 1 clusters in hypercube 45.
       > Found 1 clusters in hypercube 46.
    Cube_47 is empty.

    Cube_48 is empty.

    Cube_49 is empty.

    Cube_50 is empty.

    Cube_51 is empty.

    Cube_52 is empty.

    Cube_53 is empty.

    Cube_54 is empty.

    Cube_55 is empty.

    Cube_56 is empty.

       > Found 1 clusters in hypercube 57.
       > Found 1 clusters in hypercube 58.
       > Found 1 clusters in hypercube 59.
    Cube_60 is empty.

    Cube_61 is empty.

    Cube_62 is empty.

    Cube_63 is empty.

    Cube_64 is empty.

    Cube_65 is empty.

    Cube_66 is empty.

       > Found 1 clusters in hypercube 67.
       > Found 1 clusters in hypercube 68.
       > Found 1 clusters in hypercube 69.
    Cube_70 is empty.

    Cube_71 is empty.

    Cube_72 is empty.

    Cube_73 is empty.

    Cube_74 is empty.

    Cube_75 is empty.

    Cube_76 is empty.

    Cube_77 is empty.

    Cube_78 is empty.

    Cube_79 is empty.

    Cube_80 is empty.

    Cube_81 is empty.

    Cube_82 is empty.

    Cube_83 is empty.

    Cube_84 is empty.

    Cube_85 is empty.

       > Found 1 clusters in hypercube 86.
       > Found 1 clusters in hypercube 87.
       > Found 1 clusters in hypercube 88.
    Cube_89 is empty.

    Cube_90 is empty.

    Cube_91 is empty.

    Cube_92 is empty.

    Cube_93 is empty.

       > Found 1 clusters in hypercube 94.
       > Found 1 clusters in hypercube 95.
       > Found 1 clusters in hypercube 96.
       > Found 1 clusters in hypercube 97.
    Cube_98 is empty.

    Cube_99 is empty.

    Cube_100 is empty.

    Cube_101 is empty.

       > Found 1 clusters in hypercube 102.
    Cube_103 is empty.

    Cube_104 is empty.

    Cube_105 is empty.

       > Found 1 clusters in hypercube 106.
    Cube_107 is empty.

    Cube_108 is empty.

    Cube_109 is empty.

    Cube_110 is empty.

    Cube_111 is empty.

    Cube_112 is empty.

    Cube_113 is empty.

    Cube_114 is empty.

       > Found 1 clusters in hypercube 115.
       > Found 1 clusters in hypercube 116.
       > Found 1 clusters in hypercube 117.
    Cube_118 is empty.

    Cube_119 is empty.

    Cube_120 is empty.

    Cube_121 is empty.

    Cube_122 is empty.

       > Found 1 clusters in hypercube 123.
       > Found 1 clusters in hypercube 124.
       > Found 1 clusters in hypercube 125.
    Cube_126 is empty.

    Cube_127 is empty.

    Cube_128 is empty.

    Cube_129 is empty.

    Cube_130 is empty.

    Cube_131 is empty.

    Cube_132 is empty.

    Cube_133 is empty.

    Cube_134 is empty.

    Cube_135 is empty.

       > Found 1 clusters in hypercube 136.
       > Found 1 clusters in hypercube 137.
       > Found 1 clusters in hypercube 138.
    Cube_139 is empty.

    Cube_140 is empty.

    Cube_141 is empty.

    Cube_142 is empty.

    Cube_143 is empty.

    Cube_144 is empty.

    Cube_145 is empty.

    Cube_146 is empty.

    Cube_147 is empty.

    Cube_148 is empty.

    Cube_149 is empty.

    Cube_150 is empty.

    Cube_151 is empty.

       > Found 1 clusters in hypercube 152.
       > Found 1 clusters in hypercube 153.
       > Found 1 clusters in hypercube 154.
    Cube_155 is empty.

    Cube_156 is empty.

    Cube_157 is empty.

    Cube_158 is empty.

       > Found 1 clusters in hypercube 159.
    Cube_160 is empty.

       > Found 1 clusters in hypercube 161.
       > Found 1 clusters in hypercube 162.
    Cube_163 is empty.

    Cube_164 is empty.

    Cube_165 is empty.

    Cube_166 is empty.

    Cube_167 is empty.

    Cube_168 is empty.

    Cube_169 is empty.

    Cube_170 is empty.

    Cube_171 is empty.

    Cube_172 is empty.

       > Found 1 clusters in hypercube 173.
    Cube_174 is empty.

    Cube_175 is empty.

    Cube_176 is empty.

       > Found 1 clusters in hypercube 177.
       > Found 1 clusters in hypercube 178.
    Cube_179 is empty.

    Cube_180 is empty.

    Cube_181 is empty.

    Cube_182 is empty.

    Cube_183 is empty.

       > Found 1 clusters in hypercube 184.
       > Found 1 clusters in hypercube 185.
    Cube_186 is empty.

    Cube_187 is empty.

    Cube_188 is empty.

    Cube_189 is empty.

    Cube_190 is empty.

    Cube_191 is empty.

    Cube_192 is empty.

    Cube_193 is empty.

    Cube_194 is empty.

       > Found 1 clusters in hypercube 195.
       > Found 1 clusters in hypercube 196.
    Cube_197 is empty.

    Cube_198 is empty.

    Cube_199 is empty.

       > Found 1 clusters in hypercube 200.
       > Found 1 clusters in hypercube 201.
    Cube_202 is empty.

    Cube_203 is empty.

    Cube_204 is empty.

    Cube_205 is empty.

    Cube_206 is empty.

    Cube_207 is empty.

    Cube_208 is empty.

    Cube_209 is empty.

    Cube_210 is empty.

    Cube_211 is empty.

    Cube_212 is empty.

    Cube_213 is empty.

    Cube_214 is empty.

       > Found 1 clusters in hypercube 215.
       > Found 1 clusters in hypercube 216.
       > Found 1 clusters in hypercube 217.
       > Found 1 clusters in hypercube 218.
    Cube_219 is empty.

    Cube_220 is empty.

       > Found 1 clusters in hypercube 221.
    Cube_222 is empty.

    Cube_223 is empty.

    Cube_224 is empty.

    Cube_225 is empty.

    Cube_226 is empty.

    Cube_227 is empty.

    Cube_228 is empty.

    Cube_229 is empty.

    Cube_230 is empty.

    Cube_231 is empty.

    Cube_232 is empty.

    Cube_233 is empty.

    Cube_234 is empty.

    Cube_235 is empty.

       > Found 1 clusters in hypercube 236.
    Cube_237 is empty.

    Cube_238 is empty.

    Cube_239 is empty.

    Cube_240 is empty.

       > Found 1 clusters in hypercube 241.
    Cube_242 is empty.

    Cube_243 is empty.

    Cube_244 is empty.

       > Found 1 clusters in hypercube 245.
       > Found 1 clusters in hypercube 246.
       > Found 1 clusters in hypercube 247.
    Cube_248 is empty.

    Cube_249 is empty.

    Cube_250 is empty.

    Cube_251 is empty.

    Cube_252 is empty.

    Cube_253 is empty.

    Cube_254 is empty.

       > Found 1 clusters in hypercube 255.
       > Found 1 clusters in hypercube 256.
       > Found 1 clusters in hypercube 257.
    Cube_258 is empty.

    Cube_259 is empty.

       > Found 1 clusters in hypercube 260.
       > Found 1 clusters in hypercube 261.
    Cube_262 is empty.

    Cube_263 is empty.

       > Found 1 clusters in hypercube 264.
    Cube_265 is empty.

    Cube_266 is empty.

    Cube_267 is empty.

    Cube_268 is empty.

    Cube_269 is empty.

       > Found 1 clusters in hypercube 270.
       > Found 1 clusters in hypercube 271.
       > Found 1 clusters in hypercube 272.
    Cube_273 is empty.

    Cube_274 is empty.

    Cube_275 is empty.

    Cube_276 is empty.

    Cube_277 is empty.

    Cube_278 is empty.

    Cube_279 is empty.

       > Found 1 clusters in hypercube 280.
       > Found 1 clusters in hypercube 281.
       > Found 1 clusters in hypercube 282.
    Cube_283 is empty.

    Cube_284 is empty.

       > Found 1 clusters in hypercube 285.
       > Found 1 clusters in hypercube 286.
    Cube_287 is empty.

    Cube_288 is empty.

       > Found 1 clusters in hypercube 289.
    Cube_290 is empty.

    Cube_291 is empty.

    Cube_292 is empty.

    Cube_293 is empty.

    Cube_294 is empty.

    Cube_295 is empty.

       > Found 1 clusters in hypercube 296.
       > Found 1 clusters in hypercube 297.
       > Found 1 clusters in hypercube 298.
    Cube_299 is empty.

    Cube_300 is empty.

    Cube_301 is empty.

    Cube_302 is empty.

    Cube_303 is empty.

       > Found 1 clusters in hypercube 304.
    Cube_305 is empty.

    Cube_306 is empty.

       > Found 1 clusters in hypercube 307.
       > Found 1 clusters in hypercube 308.
       > Found 1 clusters in hypercube 309.
    Cube_310 is empty.

    Cube_311 is empty.

    Cube_312 is empty.

    Cube_313 is empty.

    Cube_314 is empty.

       > Found 1 clusters in hypercube 315.
       > Found 1 clusters in hypercube 316.
       > Found 1 clusters in hypercube 317.
    Cube_318 is empty.

    Cube_319 is empty.

    Cube_320 is empty.

    Cube_321 is empty.

    Cube_322 is empty.

    Cube_323 is empty.

    Cube_324 is empty.

    Cube_325 is empty.

    Cube_326 is empty.

    Cube_327 is empty.

    Cube_328 is empty.

    Cube_329 is empty.

    Cube_330 is empty.

       > Found 1 clusters in hypercube 331.
       > Found 1 clusters in hypercube 332.
       > Found 1 clusters in hypercube 333.
    Cube_334 is empty.

    Cube_335 is empty.

    Cube_336 is empty.

    Cube_337 is empty.

    Cube_338 is empty.

    Cube_339 is empty.

       > Found 1 clusters in hypercube 340.
    Cube_341 is empty.

    Cube_342 is empty.

    Cube_343 is empty.

    Cube_344 is empty.

    Cube_345 is empty.

    Cube_346 is empty.

       > Found 1 clusters in hypercube 347.
       > Found 1 clusters in hypercube 348.
    Cube_349 is empty.

    Cube_350 is empty.

    Cube_351 is empty.

    Cube_352 is empty.

    Cube_353 is empty.

    Cube_354 is empty.

    Cube_355 is empty.

    Cube_356 is empty.

    Cube_357 is empty.

    Cube_358 is empty.

    Cube_359 is empty.

    Cube_360 is empty.

    Cube_361 is empty.

    Cube_362 is empty.

    Cube_363 is empty.

    Cube_364 is empty.

    Cube_365 is empty.

    Cube_366 is empty.

    Cube_367 is empty.

    Cube_368 is empty.

    Cube_369 is empty.

    Cube_370 is empty.

    Cube_371 is empty.

    Cube_372 is empty.

    Cube_373 is empty.

    Cube_374 is empty.

    Cube_375 is empty.

    Cube_376 is empty.

    Cube_377 is empty.

    Cube_378 is empty.

    Cube_379 is empty.

    Cube_380 is empty.

       > Found 1 clusters in hypercube 381.
       > Found 1 clusters in hypercube 382.
       > Found 1 clusters in hypercube 383.
    Cube_384 is empty.

    Cube_385 is empty.

    Cube_386 is empty.

    Cube_387 is empty.

       > Found 1 clusters in hypercube 388.
       > Found 1 clusters in hypercube 389.
       > Found 1 clusters in hypercube 390.
       > Found 1 clusters in hypercube 391.
    Cube_392 is empty.

    Cube_393 is empty.

    Cube_394 is empty.

    Cube_395 is empty.

    Cube_396 is empty.

    Cube_397 is empty.

    Cube_398 is empty.

    Cube_399 is empty.

    Cube_400 is empty.

    Cube_401 is empty.

    Cube_402 is empty.

       > Found 1 clusters in hypercube 403.
       > Found 1 clusters in hypercube 404.
       > Found 1 clusters in hypercube 405.
       > Found 1 clusters in hypercube 406.
    Cube_407 is empty.

    Cube_408 is empty.

    Cube_409 is empty.

    Cube_410 is empty.

    Cube_411 is empty.

       > Found 1 clusters in hypercube 412.
       > Found 1 clusters in hypercube 413.
    Cube_414 is empty.

    Cube_415 is empty.

    Cube_416 is empty.

    Cube_417 is empty.

       > Found 1 clusters in hypercube 418.
       > Found 1 clusters in hypercube 419.
       > Found 1 clusters in hypercube 420.
    Cube_421 is empty.

    Cube_422 is empty.

       > Found 1 clusters in hypercube 423.
    Cube_424 is empty.

    Cube_425 is empty.

       > Found 1 clusters in hypercube 426.
       > Found 1 clusters in hypercube 427.
    Cube_428 is empty.

    Cube_429 is empty.

       > Found 1 clusters in hypercube 430.
       > Found 1 clusters in hypercube 431.
    Cube_432 is empty.

       > Found 1 clusters in hypercube 433.
       > Found 1 clusters in hypercube 434.
       > Found 1 clusters in hypercube 435.
       > Found 1 clusters in hypercube 436.
       > Found 1 clusters in hypercube 437.
    Cube_438 is empty.

    Cube_439 is empty.

    Cube_440 is empty.

       > Found 1 clusters in hypercube 441.
    Cube_442 is empty.

    Cube_443 is empty.

    Cube_444 is empty.

    Cube_445 is empty.

       > Found 1 clusters in hypercube 446.
       > Found 1 clusters in hypercube 447.
       > Found 1 clusters in hypercube 448.
       > Found 1 clusters in hypercube 449.
    Cube_450 is empty.

    Cube_451 is empty.

       > Found 1 clusters in hypercube 452.
    Cube_453 is empty.

    Cube_454 is empty.

       > Found 1 clusters in hypercube 455.
       > Found 1 clusters in hypercube 456.
    Cube_457 is empty.

    Cube_458 is empty.

    Cube_459 is empty.

    Cube_460 is empty.


    Created 296 edges and 140 nodes in 0:00:00.196013.
    Output graph examples to html
    Wrote visualization to: output/digits_custom_tooltips.html
    Wrote visualization to: output/digits_ylabel_tooltips.html




|


.. code-block:: default


    import io
    import sys
    import base64

    import matplotlib.pyplot as plt
    import numpy as np
    import sklearn
    from sklearn import datasets
    import kmapper as km

    try:
        from scipy.misc import imsave, toimage
    except ImportError as e:
        print("imsave requires you to install pillow. Run `pip install pillow` and then try again.")
        sys.exit()


    # Load digits dat
    data, labels = datasets.load_digits().data, datasets.load_digits().target

    # Create images for a custom tooltip array
    tooltip_s = []
    for image_data in data:
        output = io.BytesIO()
        img = toimage(image_data.reshape((8, 8)))  # Data was a flat row of 64 "pixels".
        img.save(output, format="PNG")
        contents = output.getvalue()
        img_encoded = base64.b64encode(contents)
        img_tag = """<img src="data:image/png;base64,{}">""".format(img_encoded.decode('utf-8'))
        tooltip_s.append(img_tag)
        output.close()

    tooltip_s = np.array(tooltip_s)  # need to make sure to feed it as a NumPy array, not a list

    # Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
    mapper = km.KeplerMapper(verbose=2)

    # Fit and transform data
    projected_data = mapper.fit_transform(data,
                                          projection=sklearn.manifold.TSNE())

    # Create the graph (we cluster on the projected data and suffer projection loss)
    graph = mapper.map(projected_data,
                       clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
                       cover=km.Cover(35, 0.4))

    # Create the visualizations (increased the graph_gravity for a tighter graph-look.)
    print("Output graph examples to html" )
    # Tooltips with image data for every cluster member
    mapper.visualize(graph,
                     title="Handwritten digits Mapper",
                     path_html="output/digits_custom_tooltips.html",
                     color_function=labels,
                     custom_tooltips=tooltip_s)
    # Tooltips with the target y-labels for every cluster member
    mapper.visualize(graph,
                     title="Handwritten digits Mapper",
                     path_html="output/digits_ylabel_tooltips.html",
                     custom_tooltips=labels)

    # Matplotlib examples
    km.draw_matplotlib(graph, layout="spring")
    plt.show()

.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  12.870 seconds)


.. _sphx_glr_download_generated_gallery_plot_digits.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_digits.py <plot_digits.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_digits.ipynb <plot_digits.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
