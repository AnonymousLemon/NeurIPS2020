function hist = plotHistory(hist,marker,color, legendText,titeName,dir_name, args)
assert(length(hist.elapsed_time) == length(hist.objVal));
assert(length(hist.gradNorm) == length(hist.props));
assert(length(hist.elapsed_time) == length(hist.testVal));
assert(length(hist.elapsed_time) == length(hist.props));

xAxisLims = [1 args.maxProps*1.0];
failed = (hist.props(end) < args.maxProps) && (hist.gradNorm(end) > args.gradTol);

hist.objVal = hist.objVal/hist.objVal(1); % Normalize so all runs start from the same point
hist.gradNorm = hist.gradNorm/hist.gradNorm(1); % Normalize so all runs start from the same point


global min_obj max_obj min_grad max_grad min_test max_test h1 h2 h3
if min_obj > min(hist.objVal)
    min_obj = min(hist.objVal);
end
if max_obj < max(hist.objVal)
    max_obj = max(hist.objVal);
end
if min_grad > min(hist.gradNorm)
    min_grad = min(hist.gradNorm);
end
if max_grad < max(hist.gradNorm)
    max_grad = max(hist.gradNorm);
end
if min_test > min(hist.testVal)
    min_test = min(hist.testVal);
end
if max_test < max(hist.testVal)
    max_test = max(hist.testVal);
end


lineWidth = 2;
lineWidthX = 2;
markerSizeX = 100;
fontSize = 12;
transparencyValue = 1;

figure(1); xlim(xAxisLims); 
if ~isempty(min_obj)  && ~isempty(max_obj)
    ylim([min_obj*0.95 max_obj*1.1]) 
end 
h = loglog(hist.props,hist.objVal,marker, 'color',color, 'LineWidth', lineWidth); 
h1(end+1) = h;
xlabel('Oracle Calls'); ylabel('\textbf{$$F(\textbf{x})$$: Objective Function}','interpreter','latex');
title(titeName);
set(gca,'fontsize',fontSize, 'fontweight','bold');
hold on;
if failed
   scatter(hist.props(end),hist.objVal(end), 'Marker', 'x', 'MarkerEdgeColor', color, 'MarkerFaceColor', color, 'MarkerEdgeAlpha', transparencyValue, 'LineWidth', lineWidthX, 'SizeData', markerSizeX);
end
legend(h1,legendText,'Location','best');
h.Color(4) = transparencyValue;

figure(2); xlim(xAxisLims); 
if ~isempty(min_grad)  && ~isempty(max_grad)
    ylim([min_grad*0.95 max_grad*1.1]); 
end
h = loglog(hist.props,hist.gradNorm,marker, 'color',color, 'LineWidth', lineWidth);
h2(end+1) = h;
xlabel('Oracle Calls'); ylabel('\textbf{$$\|\nabla F(\textbf{x})\|$$: Gradient Norm}','interpreter','latex');
title(titeName);
set(gca,'fontsize',fontSize, 'fontweight','bold');
hold on;
if failed
   scatter(hist.props(end),hist.gradNorm(end), 'Marker', 'x', 'MarkerEdgeColor', color, 'MarkerFaceColor', color, 'MarkerEdgeAlpha', transparencyValue, 'LineWidth', lineWidthX, 'SizeData', markerSizeX);
end
legend(h2,legendText,'Location','best');
h.Color(4) = transparencyValue;

figure(3); xlim(xAxisLims); 
if ~isempty(min_test) && ~isempty(max_test)
    ylim([min_test*0.95 max_test*1.1]); 
end
h = loglog(hist.props,hist.testVal,marker, 'color',color, 'LineWidth', lineWidth);
h3(end+1) = h;
xlabel('Oracle Calls'); ylabel('Test Classification Accuracy');
title(titeName);
set(gca,'fontsize',fontSize, 'fontweight','bold');
hold on;
if failed
   scatter(hist.props(end),hist.testVal(end), 'Marker', 'x', 'MarkerEdgeColor', color, 'MarkerFaceColor', color, 'MarkerEdgeAlpha', transparencyValue, 'LineWidth', lineWidthX, 'SizeData', markerSizeX);
end
legend(h3,legendText,'Location','best');
h.Color(4) = transparencyValue;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off MATLAB:legend:IgnoringExtraEntries
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(1);  
saveas(gcf,[dir_name,'/','Obj_Props'],'fig');
saveas(gcf,[dir_name,'/','Obj_Props'],'png');
saveas(gcf,[dir_name,'/','Obj_Props'],'pdf');

figure(2);  
saveas(gcf,[dir_name,'/','Grad_Props'],'fig');
saveas(gcf,[dir_name,'/','Grad_Props'],'png');
saveas(gcf,[dir_name,'/','Grad_Props'],'pdf');

figure(3);  
saveas(gcf,[dir_name,'/','Test_Props'],'fig');
saveas(gcf,[dir_name,'/','Test_Props'],'png');
saveas(gcf,[dir_name,'/','Test_Props'],'pdf');

end