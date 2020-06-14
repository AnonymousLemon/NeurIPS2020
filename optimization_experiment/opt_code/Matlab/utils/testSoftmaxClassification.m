function [accuracy, probability, predicted_labels] =  testSoftmaxClassification( test_features, test_labels, weights )
assert(length(weights) == length(shiftdim(weights))); % to make sure w is a column vector
[n,d] = size(test_features);
assert(ceil(length(weights)/d) == floor(length(weights)/d)); % to make sure that w is of a vector of length (C x d) for some integer C, where C is the number of classes.
C = length(weights)/d; % Technically the total number of classes is C+1, but the degree of freedom is only C
W = (vec2mat(weights,d))'; % A (d x C) matrix formed from w where each column is a w_b for class b, b = 1,2,C
assert(size(W,1) == d);
assert(size(W,2) == C);
assert(size(test_labels,1) == n);
assert(size(test_labels,2) == ( C + 1 ) );
XW = test_features * W;
large_vals = max(0,max(XW,[],2)); % ONLY WE CHECK FOR MAXIMUM POSTIVE VALUES!
XW_trick = XW - repmat(large_vals,1,size(W,2));
XW_1_trick = [-large_vals, XW_trick]; %To account for the "1" in the sum, i.e., exp(0)
sum_exp_trick = sum(exp(XW_1_trick),2);
inv_sum_exp = 1./sum_exp_trick;
inv_sum_exp = repmat(inv_sum_exp,1,size(W,2));
probability = inv_sum_exp.*exp(XW_trick); % (n x C) matrix
probability = [ probability (1 - sum(probability, 2)) ];
[ ~, predicted_labels] =  max( probability, [], 2 );
[ ~, actual_labels ] =  max( test_labels, [], 2 );
assert(size(actual_labels ,1) == size(predicted_labels,1));
assert(size(actual_labels ,2) == size(predicted_labels,2));
diff = actual_labels - predicted_labels; 
accuracy = (sum( diff(:) == 0 ) / size(test_labels, 1)) * 100;
end

