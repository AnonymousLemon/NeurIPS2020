function [accuracy, probability, predicted_labels] =  testNLLSClassification( test_features, test_labels, phi, weights )
assert(length(weights) == length(shiftdim(weights))); % to make sure w is a column vector
[n,d] = size(test_features);
assert(ceil(length(weights)/d) == floor(length(weights)/d)); % to make sure that w is of a vector of length (C x d) for some integer C, where C is the number of classes.
C = length(weights)/d; % Technically the total number of classes is C+1, but the degree of freedom is only C
assert(C == 1); % For NLLS
W = (vec2mat(weights,d))'; % A (d x C) matrix formed from w where each column is a w_b for class b, b = 1,2,C
assert(size(W,1) == d);
assert(size(W,2) == C);
assert(size(test_labels,1) == n);
assert(size(test_labels,2) == ( C + 1 ) );
XW = test_features * W;
probability = phi(XW);
probability = [ probability , (1 - probability) ];
[ ~, predicted_labels] =  max( probability, [], 2 );
[ ~, actual_labels ] =  max( test_labels, [], 2 );
assert(size(actual_labels ,1) == size(predicted_labels,1));
assert(size(actual_labels ,2) == size(predicted_labels,2));
diff = actual_labels - predicted_labels; 
accuracy = (sum( diff(:) == 0 ) / size(test_labels, 1)) * 100;
end

