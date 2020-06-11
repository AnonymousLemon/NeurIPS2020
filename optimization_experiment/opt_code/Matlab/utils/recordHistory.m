function hist = recordHistory(hist, fun, x, testFun)
if ~isempty(testFun)
    doTestAccuracy = true;
else
    doTestAccuracy = false;
end
endIndex = 1 + length(hist.objVal);
[f,g]=fun(x);
hist.objVal(endIndex) = f;
hist.gradNorm(endIndex) = norm(g);
if doTestAccuracy
    hist.testVal(endIndex) = testFun(x);
else
    hist.testVal(endIndex) = 0;
end
