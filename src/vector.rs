pub use std::f32::consts::PI;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

pub fn vec2(x: f32, y: f32) -> Vec2 { Vec2 { x, y } }
pub fn vec3(x: f32, y: f32, z: f32) -> Vec3 { Vec3 { x, y, z } }
pub fn vec4(x: f32, y: f32, z: f32, w: f32) -> Vec4 { Vec4 { x, y, z, w } }

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Vec2 { Vec2{x, y} }
    pub fn mul_scalar(&self, scalar: f32) -> Vec2 { Vec2::new(self.x * scalar, self.y * scalar) }
    pub fn div_scalar(&self, scalar: f32) -> Vec2 { Vec2::new(self.x / scalar, self.y / scalar) }
    pub fn magnitude(&self) -> f32 { (self.x*self.x + self.y*self.y).sqrt() }
    pub fn mag2(&self) -> f32 { self.x*self.x + self.y*self.y }
    pub fn normalize(&self) -> Vec2 { self.div_scalar(self.magnitude()) }
    pub fn lerp(&self, other: Vec2, t: f32) -> Vec2 { Vec2::new(self.x*(1.0-t) + other.x*(t), self.y*(1.0-t) + other.y*(t)) }
    pub fn dot(&self, other: Vec2) -> f32 { self.x*other.x + self.y*other.y } 
    pub fn rotate(&self, radians: f32) -> Vec2 { 
        Vec2::new(
            self.x * radians.cos() - self.y * radians.sin(), 
            self.x * radians.sin() + self.y * radians.cos()
        ) 
    }
    pub fn complex_mul(&self, other: Vec2) -> Vec2 {
        let a = self.x;
        let b = self.y;
        let c = other.x;
        let d = other.y;
        Vec2::new(a*c - b*d, a*d + c*b)
    }
    pub fn complex_div(&self, other: Vec2) -> Vec2 {
        let a = self.x;
        let b = self.y;
        let c = other.x;
        let d = other.y;

        let denom = c*c + d*d;

        Vec2::new(a*c + b*d, b*c - a*d) / denom
    }
}
impl std::fmt::Display for Vec2 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}
impl std::ops::Add<Vec2> for Vec2 {
    type Output = Vec2;

    fn add(self, _rhs: Vec2) -> Vec2 {
        Vec2 { x: self.x + _rhs.x, y: self.y + _rhs.y }
    }
}
impl std::ops::AddAssign<Vec2> for Vec2 {
    fn add_assign(&mut self, rhs: Vec2) {
        *self = *self + rhs;
    }
}
impl std::ops::SubAssign<Vec2> for Vec2 {
    fn sub_assign(&mut self, rhs: Vec2) {
        *self = *self - rhs;
    }
}
impl std::ops::Sub<Vec2> for Vec2 {
    type Output = Vec2;

    fn sub(self, _rhs: Vec2) -> Vec2 {
        Vec2 { x: self.x - _rhs.x, y: self.y - _rhs.y }
    }
}
impl std::ops::Mul<f32> for Vec2 {
    type Output = Vec2;

    fn mul(self, _rhs: f32) -> Vec2 {
        self.mul_scalar(_rhs)
    }
}
impl std::ops::Mul<Vec2> for f32 {
    type Output = Vec2;

    fn mul(self, _rhs: Vec2) -> Vec2 {
        _rhs.mul_scalar(self)
    }
}
impl std::ops::Div<f32> for Vec2 {
    type Output = Vec2;

    fn div(self, _rhs: f32) -> Vec2 {
        self.div_scalar(_rhs)
    }
}
impl std::ops::Div<Vec2> for Vec2 {
    type Output = Vec2;

    fn div(self, _rhs: Vec2) -> Vec2 {
        self.complex_div(_rhs)
    }
}
impl std::ops::Mul<Vec2> for Vec2 {
    type Output = Vec2;

    fn mul(self, _rhs: Vec2) -> Vec2 {
        self.complex_mul(_rhs)
    }
}
impl std::ops::Neg for Vec2 {
    type Output = Vec2;

    fn neg(self) -> Vec2 {
        self.mul_scalar(-1.0)
    }
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 { Vec3{x, y, z} }
    pub fn mul_scalar(&self, scalar: f32) -> Vec3 { Vec3::new(self.x * scalar, self.y * scalar, self.z * scalar) }
    pub fn div_scalar(&self, scalar: f32) -> Vec3 { Vec3::new(self.x / scalar, self.y / scalar, self.z / scalar) }
    pub fn magnitude(&self) -> f32 { (self.x*self.x + self.y*self.y + self.z*self.z).sqrt() }
    pub fn square_distance(&self) -> f32 { self.x*self.x + self.y*self.y + self.z*self.z }
    pub fn abs(&self) -> Self { vec3(self.x.abs(), self.y.abs(), self.z.abs())}
    pub fn normalize(&self) -> Vec3 { 
        let m = self.magnitude();
        if m == 0.0 {
            return vec3(0.0, 0.0, 0.0);
        } else {
            return self.div_scalar(self.magnitude()); 
        }
    }
    pub fn lerp(&self, other: Vec3, t: f32) -> Vec3 { Vec3::new(self.x*(1.0-t) + other.x*(t), self.y*(1.0-t) + other.y*(t), self.z*(1.0-t) + other.z*(t)) }
    pub fn dist(&self, other: Vec3) -> f32 {(*self - other).magnitude().sqrt()}
    pub fn dot(&self, other: Vec3) -> f32 {self.x*other.x + self.y*other.y + self.z*other.z} // is squ dist lol
    pub fn cross(&self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x,
        )
    }
    // assumes north pole
    pub fn cartesian_to_spherical(&self) -> Self {
        vec3(self.magnitude(), self.y.acos(), self.z.atan2(self.x))
    }
    pub fn spherical_to_cartesian(&self) -> Self {
        vec3(self.x*self.y.sin()*self.z.cos(), self.x*self.y.cos(), self.x*self.y.sin()*self.z.sin())
    }
    pub fn rotate_about_Vec3(&self, axis: Vec3, theta: f32) -> Vec3 {
        *self*theta.cos() + (axis.cross(*self)*theta.sin()) + axis * (axis.dot(*self)*(1.0 - theta.cos()))
    }
    pub fn xy(&self) -> Vec2 { vec2(self.x, self.y) }
    
    pub fn assert_equals(&self, other: Self) {
        if self.x - other.x != 0.0 || self.y - other.y != 0.0 || self.z - other.z != 0.0 { panic!("{} not equal to {}", self, other) }
    }
    pub fn assert_approx_equals(&self, other: Self) {
        use std::f32::EPSILON as e;
        if (self.x - other.x).abs() > e || (self.y - other.y).abs() > e || (self.z - other.z).abs() > e { panic!("{} not equal to {}", self, other); }
    }
    pub fn assert_unit(&self) {
        if self.magnitude() != 1.00 { panic!("{} not unit", self); }
    }
}

// too imprecise

#[test]
fn test_spherical() {
    ((vec3(1.0, 0.0, 0.0).cartesian_to_spherical() + vec3(0.0, PI, 0.0)).spherical_to_cartesian().assert_approx_equals(vec3(-1.0, 0.0, 0.0)));
    ((vec3(1.0, 0.0, 0.0).cartesian_to_spherical() + vec3(0.0, -PI, 0.0)).spherical_to_cartesian().assert_approx_equals(vec3(-1.0, 0.0, 0.0)));
    ((vec3(1.0, 0.0, 0.0).cartesian_to_spherical() + vec3(0.0, 2.0*PI, 0.0)).spherical_to_cartesian().assert_approx_equals(vec3(1.0, 0.0, 0.0)));
    ((vec3(1.0, 0.0, 0.0).cartesian_to_spherical() + vec3(0.0, PI, 0.0)).spherical_to_cartesian().assert_approx_equals(vec3(-1.0, 0.0, 0.0)));
}

impl std::ops::Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, _rhs: Vec3) -> Vec3 {
        Vec3 { x: self.x - _rhs.x, y: self.y - _rhs.y, z: self.z - _rhs.z }
    }
}

impl std::ops::Add<Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, _rhs: Vec3) -> Vec3 {
        Vec3 { x: self.x + _rhs.x, y: self.y + _rhs.y, z: self.z + _rhs.z}
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Vec3;

    fn mul(self, _rhs: f32) -> Vec3 {
        self.mul_scalar(_rhs)
    }
}

impl std::ops::Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, _rhs: Vec3) -> Vec3 {
        _rhs.mul_scalar(self)
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Vec3;

    fn div(self, _rhs: f32) -> Vec3 {
        self.div_scalar(_rhs)
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Vec3 {
        self.mul_scalar(-1.0)
    }
}

impl std::ops::AddAssign for Vec3 {

    fn add_assign(&mut self, rhs: Vec3) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl std::fmt::Display for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let decimals = f.precision().unwrap_or(2);
        let string = format!("[{:.*}, {:.*}, {:.*}]", decimals, self.x, decimals, self.y, decimals, self.z);
        f.pad_integral(true, "", &string)
    }
}


impl Vec4 {
    pub fn dot(&self, other: Vec4) -> f32 {
        self.x*other.x + self.y * other.y + self.z*other.z + self.w*other.w
    }
    pub fn tl(&self) -> Vec2 {vec2(self.x, self.y)}
    pub fn br(&self) -> Vec2 {vec2(self.x + self.z, self.y + self.w)}
    pub fn tr(&self) -> Vec2 {vec2(self.x + self.z, self.y)}
    pub fn bl(&self) -> Vec2 {vec2(self.x, self.y + self.w)}
    pub fn grid_child(&self, i: usize, j: usize, w: usize, h: usize) -> Vec4 {
        let cw = self.z / w as f32;
        let ch = self.w / h as f32;
        vec4(self.x + cw * i as f32, self.y + ch * j as f32, cw, ch)
    }
    pub fn hsv_to_rgb(&self) -> Vec4 {
        let v = self.z;
        let hh = (self.x % 360.0) / 60.0;
        let i = hh.floor() as i32;
        let ff = hh - i as f32;
        let p = self.z * (1.0 - self.y);
        let q = self.z * (1.0 - self.y * ff);
        let t = self.z * (1.0 - self.y * (1.0 - ff));
        match i {
            0 => vec4(v, t, p, self.w),
            1 => vec4(q, v, p, self.w),
            2 => vec4(p, v, t, self.w),
            3 => vec4(p, q, v, self.w),
            4 => vec4(t, p, v, self.w),
            5 => vec4(v, p, q, self.w),
            _ => panic!("unreachable"),
        }
    }
    fn contains(&self, p: Vec2) -> bool {
        !(p.x < self.x || p.x > self.x + self.z || p.y < self.y || p.y > self.y + self.w)
    }
    fn point_within(&self, p: Vec2) -> Vec2 {
        vec2(p.x*self.z+self.x, p.y*self.w+self.y)
    }
    fn point_without(&self, p: Vec2) -> Vec2 {
        vec2((p.x - self.x) / self.z, (p.y - self.y) / self.w)
    }
    fn fit_aspect(&self, a: f32) -> Vec4 {
        let a_self = self.z/self.w;

        if a_self > a {
            // parent wider
            vec4((self.z - self.z*(1.0/a))/2.0, 0.0, self.z*1.0/a, self.w)
        } else {
            // child wider
            vec4(0.0, (self.w - self.w*(1.0/a))/2.0, self.z, self.w*a)
        }
    }
}


