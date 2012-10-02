function create_test(fid, covf, covf_name, k)

clean_name = regexp(covf_name, '[A-Za-z0-9]+', 'match');
    
fprintf(fid, 'TEST(CovTest, %s) {\n', clean_name{1});
fprintf(fid, '  int param_dim, input_dim;\n');
fprintf(fid, '  Eigen::VectorXd p, x1, x2, g11, g12, g21, g22;\n');

for i=1:k

    D = randi(5);    
    param_dim = eval(feval(covf{:}));
    hyp = rand(1,param_dim);
    x1 = rand(1,D);
    x2 = rand(1,D);
    
    K = feval(covf{:}, hyp, [x1;x2]);
    
    fprintf(fid, '  param_dim = %d;\n', param_dim);
    fprintf(fid, '  input_dim = %d;\n', D);
    fprintf(fid, '  covf = factory.create(input_dim, "%s");\n', covf_name);
    fprintf(fid, '  p.resize(%d);\n', param_dim);
    fprintf(fid, '  x1.resize(%d);\n', D);
    fprintf(fid, '  x2.resize(%d);\n', D);
    
    fprintf(fid, '  p << %s\n', num2vectorXd(hyp));
    fprintf(fid, '  covf->set_loghyper(p);\n');
    fprintf(fid, '  x1 << %s\n', num2vectorXd(x1));
    fprintf(fid, '  x2 << %s\n', num2vectorXd(x2));
    
    fprintf(fid, '  g11.resize(param_dim);\n');
    fprintf(fid, '  g12.resize(param_dim);\n');
    fprintf(fid, '  g21.resize(param_dim);\n');
    fprintf(fid, '  g22.resize(param_dim);\n');
    fprintf(fid, '  covf->grad(x1, x1, g11);\n');
    fprintf(fid, '  covf->grad(x1, x2, g12);\n');
    fprintf(fid, '  covf->grad(x2, x1, g21);\n');
    fprintf(fid, '  covf->grad(x2, x2, g22);\n');
    
    fprintf(fid, '  ASSERT_EQ(covf->get_param_dim(), %d);\n', param_dim);
    fprintf(fid, '  ASSERT_EQ(covf->get_input_dim(), %d);\n', D);

    fprintf(fid, '  ASSERT_NEAR(%21.19f, covf->get(x1, x1), tol);\n', K(1,1));
    fprintf(fid, '  ASSERT_NEAR(%21.19f, covf->get(x2, x1), tol);\n', K(2,1));
    fprintf(fid, '  ASSERT_NEAR(%21.19f, covf->get(x1, x2), tol);\n', K(1,2));
    fprintf(fid, '  ASSERT_NEAR(%21.19f, covf->get(x2, x2), tol);\n', K(2,2));
    
    for j=1:param_dim
        g = feval(covf{:}, hyp, [x1;x2], [], j);
        fprintf(fid, '  ASSERT_NEAR(%21.19f, g11(%d), tol);\n', g(1,1), j-1);
        fprintf(fid, '  ASSERT_NEAR(%21.19f, g12(%d), tol);\n', g(1,2), j-1);
        fprintf(fid, '  ASSERT_NEAR(%21.19f, g21(%d), tol);\n', g(2,1), j-1);
        fprintf(fid, '  ASSERT_NEAR(%21.19f, g22(%d), tol);\n', g(2,2), j-1);
    end
    %  Eigen::VectorXd g1(param_dim);
    %  g1 << 0.0, 2.442805516320;
    %  Eigen::VectorXd g2(param_dim);
    %  g2 << 0.864440931000, 0.662357933089;
    
    %  test_covf(param_dim, p, K, g1, g2);
    fprintf(fid,'  delete covf;\n');
    
end

fprintf(fid,'}\n');

end

function s = num2vectorXd(x)
    
    s = num2str(x,'% 21.19f,');
    s(end) = ';';
    
end