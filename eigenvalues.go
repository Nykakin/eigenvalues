package eigenvalues

import (
	"math"
)

type EigenvalueDecomposition interface {
	Eigenvector() [][]float64
	EigenvaluesReal() []float64
	EigenvaluesImag() []float64
	BlockDiagonalEigenvalueMatrix() [][]float64
}

type eigenvalueDecomposition struct {
	n            int
	issymetric   bool
	d, e         []float64
	v, h         [][]float64
	ort          []float64
	cdivr, cdivi float64
}

func (ed eigenvalueDecomposition) Eigenvector() [][]float64 {
	return ed.v
}

func (ed eigenvalueDecomposition) EigenvaluesReal() []float64 {
	return ed.d
}

func (ed eigenvalueDecomposition) EigenvaluesImag() []float64 {
	return ed.e
}

func (ed eigenvalueDecomposition) BlockDiagonalEigenvalueMatrix() [][]float64 {
	d := make([][]float64, ed.n)
	for i := range d {
		d[i] = make([]float64, ed.n)
	}

	for i := 0; i < ed.n; i++ {
		d[i][i] = ed.d[i]
		if ed.e[i] > 0 {
			d[i][i+1] = ed.e[i]
		} else {
			d[i][i-1] = ed.e[i]
		}
	}

	return d
}

func NewEigenvalueDecomposition(matrix [][]float64) EigenvalueDecomposition {
	ed := eigenvalueDecomposition{}
	ed.n = len(matrix[0])

	ed.v = make([][]float64, ed.n)
	for i := range ed.v {
		ed.v[i] = make([]float64, ed.n)
	}
	ed.d = make([]float64, ed.n)
	ed.e = make([]float64, ed.n)

	ed.issymetric = true
	for j := 0; (j < ed.n) && ed.issymetric; j++ {
		for i := 0; (i < ed.n) && ed.issymetric; i++ {
			ed.issymetric = (matrix[i][j] == matrix[j][i])
		}
	}

	if ed.issymetric {
		for i := 0; i < ed.n; i++ {
			for j := 0; j < ed.n; j++ {
				ed.v[i][j] = matrix[i][j]
			}
		}

		// Tridiagonalize.
		ed.tred2()

		// Diagonalize.
		ed.tql2()
	} else {
		ed.h = make([][]float64, ed.n)
		for i := range ed.h {
			ed.h[i] = make([]float64, ed.n)
		}
		ed.ort = make([]float64, ed.n)
		for i := 0; i < ed.n; i++ {
			for j := 0; j < ed.n; j++ {
				ed.h[i][j] = matrix[i][j]
			}
		}

		// Reduce to Hessenberg form.
		ed.orthes()

		// Reduce Hessenberg to real Schur form.
		ed.hqr2()
	}

	return ed
}

// Symmetric Householder reduction to tridiagonal form.
func (ed eigenvalueDecomposition) tred2() {

	//  This is derived from the Algol procedures tred2 by
	//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
	//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
	//  Fortran subroutine in EISPACK.

	for j := 0; j < ed.n; j++ {
		ed.d[j] = ed.v[ed.n-1][j]
	}

	// Householder reduction to tridiagonal form.

	for i := ed.n - 1; i > 0; i-- {

		// Scale to avoid under/overflow.

		scale := 0.0
		h := 0.0
		for k := 0; k < i; k++ {
			scale = scale + math.Abs(ed.d[k])
		}
		if scale == 0.0 {
			ed.e[i] = ed.d[i-1]
			for j := 0; j < i; j++ {
				ed.d[j] = ed.v[i-1][j]
				ed.v[i][j] = 0.0
				ed.v[j][i] = 0.0
			}
		} else {

			// Generate Householder vector.

			for k := 0; k < i; k++ {
				ed.d[k] /= scale
				h += ed.d[k] * ed.d[k]
			}
			f := ed.d[i-1]
			g := math.Sqrt(h)
			if f > 0 {
				g = -g
			}
			ed.e[i] = scale * g
			h = h - f*g
			ed.d[i-1] = f - g
			for j := 0; j < i; j++ {
				ed.e[j] = 0.0
			}

			// Apply similarity transformation to remaining columns.

			for j := 0; j < i; j++ {
				f = ed.d[j]
				ed.v[j][i] = f
				g = ed.e[j] + ed.v[j][j]*f
				for k := j + 1; k <= i-1; k++ {
					g += ed.v[k][j] * ed.d[k]
					ed.e[k] += ed.v[k][j] * f
				}
				ed.e[j] = g
			}
			f = 0.0
			for j := 0; j < i; j++ {
				ed.e[j] /= h
				f += ed.e[j] * ed.d[j]
			}
			hh := f / (h + h)
			for j := 0; j < i; j++ {
				ed.e[j] -= hh * ed.d[j]
			}
			for j := 0; j < i; j++ {
				f = ed.d[j]
				g = ed.e[j]
				for k := j; k <= i-1; k++ {
					ed.v[k][j] -= (f*ed.e[k] + g*ed.d[k])
				}
				ed.d[j] = ed.v[i-1][j]
				ed.v[i][j] = 0.0
			}
		}
		ed.d[i] = h
	}

	// Accumulate transformations.

	for i := 0; i < ed.n-1; i++ {
		ed.v[ed.n-1][i] = ed.v[i][i]
		ed.v[i][i] = 1.0
		h := ed.d[i+1]
		if h != 0.0 {
			for k := 0; k <= i; k++ {
				ed.d[k] = ed.v[k][i+1] / h
			}
			for j := 0; j <= i; j++ {
				g := 0.0
				for k := 0; k <= i; k++ {
					g += ed.v[k][i+1] * ed.v[k][j]
				}
				for k := 0; k <= i; k++ {
					ed.v[k][j] -= g * ed.d[k]
				}
			}
		}
		for k := 0; k <= i; k++ {
			ed.v[k][i+1] = 0.0
		}
	}
	for j := 0; j < ed.n; j++ {
		ed.d[j] = ed.v[ed.n-1][j]
		ed.v[ed.n-1][j] = 0.0
	}
	ed.v[ed.n-1][ed.n-1] = 1.0
	ed.e[0] = 0.0
}

// Symmetric tridiagonal QL algorithm.

func (ed eigenvalueDecomposition) tql2() {

	//  This is derived from the Algol procedures tql2, by
	//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
	//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
	//  Fortran subroutine in EISPACK.

	for i := 1; i < ed.n; i++ {
		ed.e[i-1] = ed.e[i]
	}
	ed.e[ed.n-1] = 0.0

	f := 0.0
	tst1 := 0.0
	eps := math.Pow(2.0, -52.0)
	for l := 0; l < ed.n; l++ {

		// Find small subdiagonal element

		tst1 = math.Max(tst1, math.Abs(ed.d[l])+math.Abs(ed.e[l]))
		m := l
		for m < ed.n {
			if math.Abs(ed.e[m]) <= eps*tst1 {
				break
			}
			m++
		}

		// If m == l, ed.d[l] is an eigenvalue,
		// otherwise, iterate.

		if m > l {
			iter := 0
			for {
				iter = iter + 1 // (Could check iteration count here.)

				// Compute implicit shift

				g := ed.d[l]
				p := (ed.d[l+1] - g) / (2.0 * ed.e[l])
				r := math.Hypot(p, 1.0)
				if p < 0 {
					r = -r
				}
				ed.d[l] = ed.e[l] / (p + r)
				ed.d[l+1] = ed.e[l] * (p + r)
				dl1 := ed.d[l+1]
				h := g - ed.d[l]
				for i := l + 2; i < ed.n; i++ {
					ed.d[i] -= h
				}
				f = f + h

				// Implicit QL transformation.

				p = ed.d[m]
				c := 1.0
				c2 := c
				c3 := c
				el1 := ed.e[l+1]
				s := 0.0
				s2 := 0.0
				for i := m - 1; i >= l; i-- {
					c3 = c2
					c2 = c
					s2 = s
					g = c * ed.e[i]
					h = c * p
					r = math.Hypot(p, ed.e[i])
					ed.e[i+1] = s * r
					s = ed.e[i] / r
					c = p / r
					p = c*ed.d[i] - s*g
					ed.d[i+1] = h + s*(c*g+s*ed.d[i])

					// Accumulate transformation.

					for k := 0; k < ed.n; k++ {
						h = ed.v[k][i+1]
						ed.v[k][i+1] = s*ed.v[k][i] + c*h
						ed.v[k][i] = c*ed.v[k][i] - s*h
					}
				}
				p = -s * s2 * c3 * el1 * ed.e[l] / dl1
				ed.e[l] = s * p
				ed.d[l] = c * p

				// Check for convergence.

				if math.Abs(ed.e[l]) > eps*tst1 {
					break
				}
			}
		}
		ed.d[l] = ed.d[l] + f
		ed.e[l] = 0.0
	}

	// Sort eigenvalues and corresponding vectors.

	for i := 0; i < ed.n-1; i++ {
		k := i
		p := ed.d[i]
		for j := i + 1; j < ed.n; j++ {
			if ed.d[j] < p {
				k = j
				p = ed.d[j]
			}
		}
		if k != i {
			ed.d[k] = ed.d[i]
			ed.d[i] = p
			for j := 0; j < ed.n; j++ {
				p = ed.v[j][i]
				ed.v[j][i] = ed.v[j][k]
				ed.v[j][k] = p
			}
		}
	}
}

// Nonsymmetric reduction to Hessenberg form.

func (ed eigenvalueDecomposition) orthes() {

	//  This is derived from the Algol procedures orthes and ortran,
	//  by Martin and Wilkinson, Handbook for Auto. Comp.,
	//  Vol.ii-Linear Algebra, and the corresponding
	//  Fortran subroutines in EISPACK.

	low := 0
	high := ed.n - 1

	for m := low + 1; m <= high-1; m++ {

		// Scale column.

		scale := 0.0
		for i := m; i <= high; i++ {
			scale = scale + math.Abs(ed.h[i][m-1])
		}
		if scale != 0.0 {

			// Compute Householder transformation.

			h := 0.0
			for i := high; i >= m; i-- {
				ed.ort[i] = ed.h[i][m-1] / scale
				h += ed.ort[i] * ed.ort[i]
			}
			g := math.Sqrt(h)
			if ed.ort[m] > 0 {
				g = -g
			}
			h = h - ed.ort[m]*g
			ed.ort[m] = ed.ort[m] - g

			// Apply Householder similarity transformation
			// H = (I-u*u'/h)*H*(I-u*u')/h)

			for j := m; j < ed.n; j++ {
				f := 0.0
				for i := high; i >= m; i-- {
					f += ed.ort[i] * ed.h[i][j]
				}
				f = f / h
				for i := m; i <= high; i++ {
					ed.h[i][j] -= f * ed.ort[i]
				}
			}

			for i := 0; i <= high; i++ {
				f := 0.0
				for j := high; j >= m; j-- {
					f += ed.ort[j] * ed.h[i][j]
				}
				f = f / h
				for j := m; j <= high; j++ {
					ed.h[i][j] -= f * ed.ort[j]
				}
			}
			ed.ort[m] = scale * ed.ort[m]
			ed.h[m][m-1] = scale * g
		}
	}

	// Accumulate transformations (Algol's ortran).

	for i := 0; i < ed.n; i++ {
		for j := 0; j < ed.n; j++ {
			if i == j {
				ed.v[i][j] = 1.0
			} else {
				ed.v[i][j] = 0.0
			}
		}
	}

	for m := high - 1; m >= low+1; m-- {
		if ed.h[m][m-1] != 0.0 {
			for i := m + 1; i <= high; i++ {
				ed.ort[i] = ed.h[i][m-1]
			}
			for j := m; j <= high; j++ {
				g := 0.0
				for i := m; i <= high; i++ {
					g += ed.ort[i] * ed.v[i][j]
				}
				// Double division avoids possible underflow
				g = (g / ed.ort[m]) / ed.h[m][m-1]
				for i := m; i <= high; i++ {
					ed.v[i][j] += g * ed.ort[i]
				}
			}
		}
	}
}

// Complex scalar division.

func (ed eigenvalueDecomposition) cdiv(xr, xi, yr, yi float64) {
	r := 0.0
	d := 0.0
	if math.Abs(yr) > math.Abs(yi) {
		r = yi / yr
		d = yr + r*yi
		ed.cdivr = (xr + r*xi) / d
		ed.cdivi = (xi - r*xr) / d
	} else {
		r = yr / yi
		d = yi + r*yr
		ed.cdivr = (r*xr + xi) / d
		ed.cdivi = (r*xi - xr) / d
	}
}

// Nonsymmetric reduction from Hessenberg to real Schur form.

func (ed eigenvalueDecomposition) hqr2() {

	//  This is derived from the Algol procedure hqr2,
	//  by Martin and Wilkinson, Handbook for Auto. Comp.,
	//  Vol.ii-Linear Algebra, and the corresponding
	//  Fortran subroutine in EISPACK.

	// Initialize

	nn := ed.n
	n := nn - 1
	low := 0
	high := nn - 1
	eps := math.Pow(2.0, -52.0)
	exshift := 0.0

	var p, q, r, s, z, t, w, x, y float64

	// Store roots isolated by balanc and compute matrix norm

	norm := 0.0
	for i := 0; i < nn; i++ {
		if i < low || i > high {
			ed.d[i] = ed.h[i][i]
			ed.e[i] = 0.0
		}
		for j := int(math.Max(float64(i-1), 0)); j < nn; j++ {
			norm = norm + math.Abs(ed.h[i][j])
		}
	}

	// Outer loop over eigenvalue index

	iter := 0
	for n >= low {

		// Look for single small sub-diagonal element

		l := n
		for l > low {
			s = math.Abs(ed.h[l-1][l-1]) + math.Abs(ed.h[l][l])
			if s == 0.0 {
				s = norm
			}
			if math.Abs(ed.h[l][l-1]) < eps*s {
				break
			}
			l--
		}

		// Check for convergence
		// One root found

		if l == n {
			ed.h[n][n] = ed.h[n][n] + exshift
			ed.d[n] = ed.h[n][n]
			ed.e[n] = 0.0
			n--
			iter = 0

			// Two roots found

		} else if l == n-1 {
			w = ed.h[n][n-1] * ed.h[n-1][n]
			p = (ed.h[n-1][n-1] - ed.h[n][n]) / 2.0
			q = p*p + w
			z = math.Sqrt(math.Abs(q))
			ed.h[n][n] = ed.h[n][n] + exshift
			ed.h[n-1][n-1] = ed.h[n-1][n-1] + exshift
			x = ed.h[n][n]

			// Real pair

			if q >= 0 {
				if p >= 0 {
					z = p + z
				} else {
					z = p - z
				}
				ed.d[n-1] = x + z
				ed.d[n] = ed.d[n-1]
				if z != 0.0 {
					ed.d[n] = x - w/z
				}
				ed.e[n-1] = 0.0
				ed.e[n] = 0.0
				x = ed.h[n][n-1]
				s = math.Abs(x) + math.Abs(z)
				p = x / s
				q = z / s
				r = math.Sqrt(p*p + q*q)
				p = p / r
				q = q / r

				// Row modification

				for j := n - 1; j < nn; j++ {
					z = ed.h[n-1][j]
					ed.h[n-1][j] = q*z + p*ed.h[n][j]
					ed.h[n][j] = q*ed.h[n][j] - p*z
				}

				// Column modification

				for i := 0; i <= n; i++ {
					z = ed.h[i][n-1]
					ed.h[i][n-1] = q*z + p*ed.h[i][n]
					ed.h[i][n] = q*ed.h[i][n] - p*z
				}

				// Accumulate transformations

				for i := low; i <= high; i++ {
					z = ed.v[i][n-1]
					ed.v[i][n-1] = q*z + p*ed.v[i][n]
					ed.v[i][n] = q*ed.v[i][n] - p*z
				}

				// Complex pair

			} else {
				ed.d[n-1] = x + p
				ed.d[n] = x + p
				ed.e[n-1] = z
				ed.e[n] = -z
			}
			n = n - 2
			iter = 0

			// No convergence yet

		} else {

			// Form shift

			x = ed.h[n][n]
			y = 0.0
			w = 0.0
			if l < n {
				y = ed.h[n-1][n-1]
				w = ed.h[n][n-1] * ed.h[n-1][n]
			}

			// Wilkinson's original ad hoc shift

			if iter == 10 {
				exshift += x
				for i := low; i <= n; i++ {
					ed.h[i][i] -= x
				}
				s = math.Abs(ed.h[n][n-1]) + math.Abs(ed.h[n-1][n-2])
				y = 0.75 * s
				x = y
				w = -0.4375 * s * s
			}

			// MATLAB's new ad hoc shift

			if iter == 30 {
				s = (y - x) / 2.0
				s = s*s + w
				if s > 0 {
					s = math.Sqrt(s)
					if y < x {
						s = -s
					}
					s = x - w/((y-x)/2.0+s)
					for i := low; i <= n; i++ {
						ed.h[i][i] -= s
					}
					exshift += s
					w = 0.964
					x = w
					y = w
				}
			}

			iter = iter + 1 // (Could check iteration count here.)

			// Look for two consecutive small sub-diagonal elements

			m := n - 2
			for m >= l {
				z = ed.h[m][m]
				r = x - z
				s = y - z
				p = (r*s-w)/ed.h[m+1][m] + ed.h[m][m+1]
				q = ed.h[m+1][m+1] - z - r - s
				r = ed.h[m+2][m+1]
				s = math.Abs(p) + math.Abs(q) + math.Abs(r)
				p = p / s
				q = q / s
				r = r / s
				if m == l {
					break
				}
				if math.Abs(ed.h[m][m-1])*(math.Abs(q)+math.Abs(r)) <
					eps*(math.Abs(p)*(math.Abs(ed.h[m-1][m-1])+math.Abs(z)+
						math.Abs(ed.h[m+1][m+1]))) {
					break
				}
				m--
			}

			for i := m + 2; i <= n; i++ {
				ed.h[i][i-2] = 0.0
				if i > m+2 {
					ed.h[i][i-3] = 0.0
				}
			}

			// Double QR step involving rows l:n and columns m:n

			for k := m; k <= n-1; k++ {
				notlast := (k != n-1)
				if k != m {
					p = ed.h[k][k-1]
					q = ed.h[k+1][k-1]
					if notlast {
						r = ed.h[k+2][k-1]
					} else {
						r = 0.0
					}
					x = math.Abs(p) + math.Abs(q) + math.Abs(r)
					if x != 0.0 {
						p = p / x
						q = q / x
						r = r / x
					}
				}
				if x == 0.0 {
					break
				}
				s = math.Sqrt(p*p + q*q + r*r)
				if p < 0 {
					s = -s
				}
				if s != 0 {
					if k != m {
						ed.h[k][k-1] = -s * x
					} else if l != m {
						ed.h[k][k-1] = -ed.h[k][k-1]
					}
					p = p + s
					x = p / s
					y = q / s
					z = r / s
					q = q / p
					r = r / p

					// Row modification

					for j := k; j < nn; j++ {
						p = ed.h[k][j] + q*ed.h[k+1][j]
						if notlast {
							p = p + r*ed.h[k+2][j]
							ed.h[k+2][j] = ed.h[k+2][j] - p*z
						}
						ed.h[k][j] = ed.h[k][j] - p*x
						ed.h[k+1][j] = ed.h[k+1][j] - p*y
					}

					// Column modification

					for i := 0; i <= int(math.Min(float64(n), float64(k+3))); i++ {
						p = x*ed.h[i][k] + y*ed.h[i][k+1]
						if notlast {
							p = p + z*ed.h[i][k+2]
							ed.h[i][k+2] = ed.h[i][k+2] - p*r
						}
						ed.h[i][k] = ed.h[i][k] - p
						ed.h[i][k+1] = ed.h[i][k+1] - p*q
					}

					// Accumulate transformations

					for i := low; i <= high; i++ {
						p = x*ed.v[i][k] + y*ed.v[i][k+1]
						if notlast {
							p = p + z*ed.v[i][k+2]
							ed.v[i][k+2] = ed.v[i][k+2] - p*r
						}
						ed.v[i][k] = ed.v[i][k] - p
						ed.v[i][k+1] = ed.v[i][k+1] - p*q
					}
				} // (s != 0)
			} // k loop
		} // check convergence
	} // while (n >= low)

	// Backsubstitute to find vectors of upper triangular form

	if norm == 0.0 {
		return
	}

	for n := nn - 1; n >= 0; n-- {
		p = ed.d[n]
		q = ed.e[n]

		// Real vector

		if q == 0 {
			l := n
			ed.h[n][n] = 1.0
			for i := n - 1; i >= 0; i-- {
				w = ed.h[i][i] - p
				r = 0.0
				for j := l; j <= n; j++ {
					r = r + ed.h[i][j]*ed.h[j][n]
				}
				if ed.e[i] < 0.0 {
					z = w
					s = r
				} else {
					l = i
					if ed.e[i] == 0.0 {
						if w != 0.0 {
							ed.h[i][n] = -r / w
						} else {
							ed.h[i][n] = -r / (eps * norm)
						}

						// Solve real equations

					} else {
						x = ed.h[i][i+1]
						y = ed.h[i+1][i]
						q = (ed.d[i]-p)*(ed.d[i]-p) + ed.e[i]*ed.e[i]
						t = (x*s - z*r) / q
						ed.h[i][n] = t
						if math.Abs(x) > math.Abs(z) {
							ed.h[i+1][n] = (-r - w*t) / x
						} else {
							ed.h[i+1][n] = (-s - y*t) / z
						}
					}

					// Overflow control

					t = math.Abs(ed.h[i][n])
					if (eps*t)*t > 1 {
						for j := i; j <= n; j++ {
							ed.h[j][n] = ed.h[j][n] / t
						}
					}
				}
			}

			// Complex vector

		} else if q < 0 {
			l := n - 1

			// Last vector component imaginary so matrix is triangular

			if math.Abs(ed.h[n][n-1]) > math.Abs(ed.h[n-1][n]) {
				ed.h[n-1][n-1] = q / ed.h[n][n-1]
				ed.h[n-1][n] = -(ed.h[n][n] - p) / ed.h[n][n-1]
			} else {
				ed.cdiv(0.0, -ed.h[n-1][n], ed.h[n-1][n-1]-p, q)
				ed.h[n-1][n-1] = ed.cdivr
				ed.h[n-1][n] = ed.cdivi
			}
			ed.h[n][n-1] = 0.0
			ed.h[n][n] = 1.0
			for i := n - 2; i >= 0; i-- {
				var ra, sa, vr, vi float64
				ra = 0.0
				sa = 0.0
				for j := l; j <= n; j++ {
					ra = ra + ed.h[i][j]*ed.h[j][n-1]
					sa = sa + ed.h[i][j]*ed.h[j][n]
				}
				w = ed.h[i][i] - p

				if ed.e[i] < 0.0 {
					z = w
					r = ra
					s = sa
				} else {
					l = i
					if ed.e[i] == 0 {
						ed.cdiv(-ra, -sa, w, q)
						ed.h[i][n-1] = ed.cdivr
						ed.h[i][n] = ed.cdivi
					} else {

						// Solve complex equations

						x = ed.h[i][i+1]
						y = ed.h[i+1][i]
						vr = (ed.d[i]-p)*(ed.d[i]-p) + ed.e[i]*ed.e[i] - q*q
						vi = (ed.d[i] - p) * 2.0 * q
						if vr == 0.0 && vi == 0.0 {
							vr = eps * norm * (math.Abs(w) + math.Abs(q) +
								math.Abs(x) + math.Abs(y) + math.Abs(z))
						}
						ed.cdiv(x*r-z*ra+q*sa, x*s-z*sa-q*ra, vr, vi)
						ed.h[i][n-1] = ed.cdivr
						ed.h[i][n] = ed.cdivi
						if math.Abs(x) > (math.Abs(z) + math.Abs(q)) {
							ed.h[i+1][n-1] = (-ra - w*ed.h[i][n-1] + q*ed.h[i][n]) / x
							ed.h[i+1][n] = (-sa - w*ed.h[i][n] - q*ed.h[i][n-1]) / x
						} else {
							ed.cdiv(-r-y*ed.h[i][n-1], -s-y*ed.h[i][n], z, q)
							ed.h[i+1][n-1] = ed.cdivr
							ed.h[i+1][n] = ed.cdivi
						}
					}

					// Overflow control

					t = math.Max(math.Abs(ed.h[i][n-1]), math.Abs(ed.h[i][n]))
					if (eps*t)*t > 1 {
						for j := i; j <= n; j++ {
							ed.h[j][n-1] = ed.h[j][n-1] / t
							ed.h[j][n] = ed.h[j][n] / t
						}
					}
				}
			}
		}
	}

	// Vectors of isolated roots

	for i := 0; i < nn; i++ {
		if i < low || i > high {
			for j := i; j < nn; j++ {
				ed.v[i][j] = ed.h[i][j]
			}
		}
	}

	// Back transformation to get eigenvectors of original matrix

	for j := nn - 1; j >= low; j-- {
		for i := low; i <= high; i++ {
			z = 0.0
			for k := low; k <= int(math.Min(float64(j), float64(high))); k++ {
				z = z + ed.v[i][k]*ed.h[k][j]
			}
			ed.v[i][j] = z
		}
	}
}
