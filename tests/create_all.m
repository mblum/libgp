rng(2387);

fid = fopen('cov_test.cc', 'w');

fprintf(fid, '// libgp - Gaussian process library for Machine Learning\n');
fprintf(fid, '// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>\n');
fprintf(fid, '// All rights reserved.\n\n');

fprintf(fid, '#include "cov_factory.h"\n\n');

fprintf(fid, '#include <Eigen/Dense>\n');
fprintf(fid, '#include <gtest/gtest.h>\n\n');

fprintf(fid, 'const double tol = 10e-12;\n\n');

fprintf(fid, 'libgp::CovFactory factory;\n');
fprintf(fid, 'libgp::CovarianceFunction * covf;\n\n');

n=10;

create_test(fid, {'covLINard'}, 'CovLinearard', n)
create_test(fid, {'covLINone'}, 'CovLinearone', n)
create_test(fid, {'covMaterniso',3}, 'CovMatern3iso', n)
create_test(fid, {'covMaterniso',5}, 'CovMatern5iso', n)
%create_test(fid, {'covSEiso'}, 'CovRBFCS', n)
create_test(fid, {'covRQiso'}, 'CovRQiso', n)
create_test(fid, {'covSEard'}, 'CovSEard', n)
create_test(fid, {'covSEiso'}, 'CovSEiso', n)
create_test(fid, {'covNoise'}, 'CovNoise', n)
create_test(fid, {'covSum', {'covSEiso','covNoise'}}, 'CovSum(CovSEiso, CovNoise)', n)

fclose(fid);
