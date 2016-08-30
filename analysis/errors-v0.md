## Observations

### Learning type of questions.

In general, the model learns the type of questions well. For example, `which venue` -> `miami's sun life stadium`, `what year` -> `1979`.


## Errors

### Out-of-context questions

```
[f1] 0
[id] 56d98db6dc89441400fdb554
[question] what is the name of the stadium where super bowl 50 was played?
[u"levi's stadium", u"levi's stadium", u"levi's stadium"]
[prediction] new orleans ' mercedes-benz superdome
[context] the league eventually narrowed the bids to three sites: new orleans' mercedes-benz superdome, miami's sun life stadium, and the san francisco bay area's levi's stadium.
```

### Common-sense knowledge

```
[f1] 0
[id] 56be5438acb8001400a5031c
[question] which california venue was one of three considered for super bowl 50?
[u"san francisco bay area's levi's stadium",
 u"san francisco bay area's levi's stadium",
 u"levi's stadium"]
[prediction] new orleans ' mercedes-benz superdome
[context] the league eventually narrowed the bids to three sites: new orleans' mercedes-benz superdome, miami's sun life stadium, and the san francisco bay area's levi's stadium.
```

```
[f1] 0
[id] 56be5438acb8001400a5031b
[question] which florida venue was one of three considered for super bowl 50?
[u"miami's sun life stadium", u"miami's sun life stadium", u'sun life stadium']
[prediction] new orleans ' mercedes-benz superdome
[context] the league eventually narrowed the bids to three sites: new orleans' mercedes-benz superdome, miami's sun life stadium, and the san francisco bay area's levi's stadium.
```

This can be solved by matching for example `california` to `san francisco bay area`.

### Errors caused by tokenization.

```
[f1] 0
[id] 57265ceddd62a815002e82b9
[question] by which year did chrysler ended its full sized luxury model?
[u'1981', u'1981', u'1981', u'1981', u'1981']
[prediction] 1979
[context] federal safety standards, such as nhtsa federal motor vehicle safety standard 215 (pertaining to safety bumpers), and compacts like the 1974 mustang i were a prelude to the dot "downsize" revision of vehicle categories. by 1977, gm's full-sized cars reflected the crisis. by 1979, virtually all "full-size" american cars had shrunk, featuring smaller engines and smaller outside dimensions. chrysler ended production of their full-sized luxury sedans at the end of the 1981 model year, moving instead to a full front-wheel drive lineup for 1982 (except for the m-body dodge diplomat/plymouth gran fury and chrysler new yorker fifth avenue sedans).
```
Full-sized vs. Full size.

### ?

It appears that for a lot of questions, LSTM matches the span with keywords in the sentence.

```
[f1] 0
[id] 56e10e73cd28a01900c674ec
[question] what was he studying that gave him the teleforce weapon idea?
[u'van de graaff generator',
 u'van de graaff generator',
 u'van de graaff generator']
[prediction] teleforce
[context] later in life, tesla made claims concerning a "teleforce" weapon after studying the van de graaff generator. the press variably referred to it as a "peace ray" or death ray. tesla described the weapon as capable of being used against ground-based infantry or for anti-aircraft purposes.
```

