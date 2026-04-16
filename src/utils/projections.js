const EPS = 1e-10;

function assertVector(vector, name) {
  if (!vector || typeof vector.length !== "number") {
    throw new Error(`${name} must be an array-like vector.`);
  }
  if (vector.length === 0) {
    throw new Error(`${name} must not be empty.`);
  }
}

function assertSameDimensions(vectors) {
  const dim = vectors[0].length;
  for (let i = 1; i < vectors.length; i++) {
    if (vectors[i].length !== dim) {
      throw new Error("All vectors must have the same dimension.");
    }
  }
}

function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) total += a[i] * b[i];
  return total;
}

function norm(vector) {
  return Math.sqrt(dot(vector, vector));
}

function subtractProjection(vector, unitBasis) {
  const out = Float64Array.from(vector);
  for (const basisVector of unitBasis) {
    const amount = dot(out, basisVector);
    for (let i = 0; i < out.length; i++) out[i] -= amount * basisVector[i];
  }
  return out;
}

function buildOrthonormalBasis(vectors) {
  const basis = [];
  for (const vector of vectors) {
    const residual = subtractProjection(vector, basis);
    const residualNorm = norm(residual);
    if (residualNorm <= EPS) continue;

    const unit = new Float64Array(residual.length);
    for (let i = 0; i < residual.length; i++) unit[i] = residual[i] / residualNorm;
    basis.push(unit);
  }
  return basis;
}

function validateVectors(vectors) {
  vectors.forEach((vector, index) => assertVector(vector, `vector ${index + 1}`));
  assertSameDimensions(vectors);
}

/**
 * Project one n-dimensional vector onto the z axis: (0, 0, z).
 * The projected vector has the same length as the source vector.
 */
export function projectOneVector(vector) {
  validateVectors([vector]);
  return [0, 0, norm(vector)];
}

/**
 * Project two n-dimensional vectors isometrically onto the yz plane: (0, y, z).
 * The projected vectors preserve source lengths and their pairwise angle.
 */
export function projectTwoVectors(vectorA, vectorB) {
  validateVectors([vectorA, vectorB]);

  const lengthA = norm(vectorA);
  const lengthB = norm(vectorB);

  if (lengthA <= EPS) {
    return [
      [0, 0, 0],
      [0, 0, lengthB],
    ];
  }

  const zB = dot(vectorA, vectorB) / lengthA;
  const yB = Math.sqrt(Math.max(0, lengthB * lengthB - zB * zB));

  return [
    [0, 0, lengthA],
    [0, yB, zB],
  ];
}

/**
 * Project three n-dimensional vectors isometrically into R^3.
 * The projected vectors preserve source lengths and all pairwise angles.
 */
export function projectThreeVectors(vectorA, vectorB, vectorC) {
  const vectors = [vectorA, vectorB, vectorC];
  validateVectors(vectors);

  const basis = buildOrthonormalBasis(vectors);
  return vectors.map((vector) => {
    const z = basis[0] ? dot(vector, basis[0]) : 0;
    const y = basis[1] ? dot(vector, basis[1]) : 0;
    const x = basis[2] ? dot(vector, basis[2]) : 0;
    return [x, y, z];
  });
}

