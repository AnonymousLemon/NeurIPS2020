function error = testGMMEstimation(x,phi,x_true,alpha_true)
alpha = phi(x(1));
x = x(2:end);
x_true = [x_true(1,:)';x_true(2,:)'];
error = ( abs(alpha - alpha_true)/alpha_true + norm(x-x_true)/norm(x_true) ) /2;
end