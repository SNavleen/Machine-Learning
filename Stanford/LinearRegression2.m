X = [1, 5, 6, 3;
     1, 9, 10, 7;
     1, 3, 10, 6;
     1, 4, 5, 2];
Y = [100;
     430;
     325;
     115];

global zero = [0; 
               0;
               0;
               0];

theta = zero;


%X = [1, 5;
%     1, 9;
%     1, 3;
%     1, 4];
%Y = [10;
%     43;
%     32;
%     11];

%global zero = [0; 
%               0];

%theta = zero;

alpha = 0.01;

function h = getH(theta, X)
  h = theta' * X;
end

function cost = getCost(theta, X, Y)
  m = length(X);
  cost = 0;

  for i = 1:m
    prediction = getH(theta, X(i, :)');
    diff = prediction - Y(i);
    cost = cost + diff * diff;
  end
  cost = cost / (2 * m);
end

function newTheta = gradientDescent(theta, alpha, X, Y)
  global zero;
  newTheta = zero;
  cost = zero;

  m = length(X);
  n = length(theta);

  for j = 1:n
    for i = 1:m
      cost(j) += ((getH(theta, X(i, :)') - Y(i)) * X(i, j));
    end
  end

  for j = 1:n
    newTheta(j) = theta(j) - ((alpha * cost(j)) / m);
  end
end

function newTheta = normalization(theta, X, Y)
  newTheta = pinv(X' * X) * X' * Y;
end

%figure;
%hold on

%for i = 1:1000
%  theta = gradientDescent(theta, alpha, X, Y);
%  if mod(i, 100) == 0
%    disp(getCost(theta, X, Y));
%    plot(i, getCost(theta, X, Y), 'rx');
%  end
%end

%hold off

%disp(theta);
theta = normalization(theta, X, Y);
disp(theta);

%disp([theta, cost(theta, x, y)]);

%disp(theta);
%disp("\n");
%disp(X(1, :));
%disp(getH(theta, X(1, :)'));

% plot the data and the best-fit line
%figure;
%plot(X, Y, 'rx');
%hold on;
%plot(X(:, 2), X * theta, '-');
%plot(X(:, 3), X * theta, '-');
%plot(X(:, 4), X * theta, '-');
%hold off