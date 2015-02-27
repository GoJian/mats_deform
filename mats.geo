es_fluid    = 0.2;
es_mat      = 0.2;
es_boundary = 0.05;
Point(1) = {1, 1, 0, es_fluid};
Point(2) = {-1, -1, 0, es_mat};
Point(3) = {1, -1, 0, es_mat};
Point(4) = {-1, 1, 0, es_fluid};
Point(5) = {-0.5, -0.5, 0, es_boundary};
Point(6) = {0, -0.5, 0, es_mat};
Point(8) = {-1, -0.5, 0, es_boundary};
Point(9) = {1, -0.5, 0, es_boundary};
Point(10) = {0.5, -0.5, 0, es_boundary};
Line(1) = {4, 8};
Line(2) = {8, 2};
Line(3) = {2, 3};
Line(4) = {3, 9};
Line(5) = {9, 1};
Line(6) = {1, 4};
Line(7) = {8, 5};
Line(8) = {10, 9};
Circle(9) = {10, 6, 5};
Line Loop(11) = {6, 1, 7, -9, 8, 5};
Plane Surface(11) = {11};
Line Loop(13) = {3, 4, -8, 9, -7, 2};
Plane Surface(13) = {13};
Physical Line(14) = {1, 2};  // Left
Physical Line(15) = {6};     // Bottom
Physical Line(16) = {4, 5};  // Right
Physical Line(17) = {3};     // Top
Physical Surface("fluid") = {11};
Physical Surface("mat") = {13};

Mesh.Algorithm = 8;
Mesh.RecombineAll = 1;