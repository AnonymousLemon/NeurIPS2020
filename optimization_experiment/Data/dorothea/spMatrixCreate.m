function A = spMatrixCreate(X)
A = sparse(0);
row = 1;
rptNaN = 0;
prevIndex = 0;
for i=1:size(X,1)
    i
    for j = 1:size(X,2)
        if X(i,j) == X(i,j) % to find NaN which signals a new row
            if X(i,j) < prevIndex
                row = row + 1;
            end
            rptNaN = 0;
            A(row, X(i,j)) = 1;
            prevIndex = X(i,j);
        else
            rptNaN = rptNaN + 1;
            if rptNaN == 1
                row = row + 1;
                prevIndex = 0;
            end
        end
    end
end

end

% A = sparse(m,n);
% row = 1;
% rptNaN = 0;
% prevIndex = 0;
% for i=1:size(X,1)
%     i
%     for j = 1:size(X,2)
%         if X(i,j) == X(i,j) % to find NaN which signals a new row
%             if X(i,j) < prevIndex
%                 row = row + 1;
%             end
%             rptNaN = 0;
%             A(row, X(i,j)) = 1;
%             prevIndex = X(i,j);
%         else
%             rptNaN = rptNaN + 1;
%             if rptNaN == 1
%                 row = row + 1;
%                 prevIndex = 0;
%             end
%         end
%     end
% end
