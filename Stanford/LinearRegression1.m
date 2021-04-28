x = [1.1,1.3,1.5,2,2.2,2.9,3,3.2,3.2,3.7,3.9,4,4,4.1,4.5,4.9,5.1,5.3,5.9,6,6.8,7.1,7.9,8.2,8.7,9,9.5,9.6,10.3,10.5];
y = [39343,46205,37731,43525,39891,56642,60150,54445,64445,57189,63218,55794,56957,57081,61111,67938,66029,83088,81363,93940,91738,98273,101302,113812,109431,105582,116969,112635,122391,121872];

theta = [0, 0];
alpha = 0.01;

function prediction = h(theta, x)
  prediction = theta(1) + theta(2) * x;
end

function error = cost(theta, x, y)
  m = length(x);
  error = 0;

  for i = 1:m
    prediction = h(theta, x(i));
    diff = prediction - y(i);
    error = error + diff * diff;
  end
  error = error / (2 * m);
end

function newTheta = gradientDescent(theta, alpha, x, y)
  m = length(x);
  newTheta = [0, 0];
  error = [0, 0];

  for i = 1:m
    error(1) = error(1) + (h(theta, x(i)) - y(i));
    error(2) = error(2) + ((h(theta, x(i)) - y(i)) * x(i));
  end
  newTheta(1) = theta(1) - ((alpha * error(1)) / m);
  newTheta(2) = theta(2) - ((alpha * error(2)) / m);
end


figure;
hold on

for i = 1:3000
  theta = gradientDescent(theta, alpha, x, y);
  if mod(i, 100) == 0
    disp(cost(theta, x, y));
    plot(i, cost(theta, x, y), 'bx');
  end
end

hold off

% plot the data and the best-fit line
%figure;
%set (0,'defaultaxesposition', [0.15, 0.1, 0.7, 0.7]);

%plot(x, y, 'rx');
%hold on;
%plot(1, h(theta, 1));
%plot(min(x):max(x), h(theta, min(x):max(x)), '-')
%plot(X(:, 2), X * theta, '-')