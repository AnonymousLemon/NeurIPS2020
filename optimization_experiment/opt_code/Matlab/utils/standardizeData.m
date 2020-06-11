function standardizedData = standardizeData(data)
fprintf('Standardizing Data...');
standardizedData = sparse(size(data,1),size(data,2));
feature_mean = mean ( data, 1);    
feature_std  = std ( data, [], 1 );
bad_features = find(feature_std == 0);
good_features = find(feature_std ~= 0);
assert(isempty(intersect(bad_features,good_features)));
ss = ones ( size(data,1), 1 );
% standardizedData(:,good_features) = (data(:,good_features) - gmultiply(ss, feature_mean(good_features))) ./ gmultiply(ss, feature_std(good_features));
standardizedData(:,good_features) = (data(:,good_features) - (ss * feature_mean(good_features))) ./ (ss * feature_std(good_features));
standardizedData(:,bad_features) = data(:,bad_features);
assert(sum(sum(standardizedData ~= standardizedData))==0 ); % check for NaN
assert( max( abs( mean ( standardizedData, 1) ) )  <= 1E-10); % check for mean zero
assert(max( abs( std ( standardizedData, [], 1 ) - 1 ) ) <= 1E-10); % check for std one 
assert(min( abs( std ( standardizedData, [], 1 ) - 1 ) ) >= 0); % check for std one 
fprintf('Done\n');
end