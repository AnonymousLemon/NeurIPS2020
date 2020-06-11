function normalizedData = normalizeData(data)
fprintf('Normalizing Data...');
normalizedData = sparse(size(data,1),size(data,2));
feature_norms = sqrt( sum( data.*data , 1 ) );
bad_features = find(feature_norms == 0);
good_features = find(feature_norms ~= 0);
assert(isempty(intersect(bad_features,good_features)));
ss = ones ( size(data,1), length(good_features) );
normalizedData(:,good_features) = data(:,good_features) ./ gmultiply(ss, feature_norms(good_features));
normalizedData(:,bad_features) = data(:,bad_features);
assert(sum(sum(normalizedData ~= normalizedData))==0 ); % check for NaN
%assert(abs(sum(sum(normalizedData.*normalizedData,1)) - size(normalizedData,2))/size(normalizedData,2) <= 1E-10);
fprintf('Done\n');
end