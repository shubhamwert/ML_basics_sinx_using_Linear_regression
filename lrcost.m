function J = lrcost(X, y, theta)
m=length(y);
temp=(X*theta-y).^2;
J=sum(temp)/(2*m);

length(J);


end
