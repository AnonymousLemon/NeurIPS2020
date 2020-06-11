function[X,Y] = sortAndScaleData(X,labels)
%[X,Y] = sortAndScaleData(X,labels)
%

% Scale X [-0.5 0.5]
%X  = X/max(abs(X(:))) - 0.5;
shift = max(0,max(X(:)));
X = X - shift;
% Organize labels
[~,k] = sort(labels);
labels = labels(k);
X      = X(k,:);

Y = zeros(size(X,1),max(labels)-min(labels)+1);
for i=1:size(X,1)
    Y(i,labels(i)+1) = 1;
end
