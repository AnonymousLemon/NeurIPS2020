function accuracy =  testRegressionResult( test_X, test_y, weights )
assert(length(weights) == length(shiftdim(weights))); % to make sure w is a column vector
assert(length(test_y) == length(shiftdim(test_y))); % to make sure y is a column vector
[n,d] = size(test_X);
assert(length(weights) == d);
assert(length(test_y) == size(test_X,1));
assert(size(test_y,1) == n);
accuracy = 0.5*norm(test_X*weights - test_y)^2;
end

