use std::path::Path;
use std::fs::File;
use std::io::BufWriter;
use crate::vector::*;

pub struct ImageBuffer {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl ImageBuffer {
    pub fn new(w: usize, h: usize) -> Self {
        ImageBuffer {
            w,
            h,
            data: vec![0; w*h*4],
        }
    }
    pub fn from_bytes(png_bytes: &[u8]) -> Self {
        let decoder = png::Decoder::new(png_bytes);
        let mut reader = decoder.read_info().unwrap();
        let mut data = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut data).unwrap();
        data.truncate(info.buffer_size());
        ImageBuffer {
            w: info.width as usize,
            h: info.height as usize,
            data,
        }
    }
    pub fn get(&self, x: usize, y: usize) -> Vec4 {
        let idx = (y*self.w + x)*4;
        vec4(
            self.data[idx] as f32 / 255.0,
            self.data[idx+1] as f32 / 255.0,
            self.data[idx+2] as f32 / 255.0,
            self.data[idx+3] as f32 / 255.0,
        )
    }
    pub fn set(&mut self, x: usize, y: usize, colour: Vec4) {
        let idx = (y*self.w + x)*4;
        self.data[idx] = (colour.x * 255.0) as u8;
        self.data[idx + 1] = (colour.y * 255.0) as u8;
        self.data[idx + 2] = (colour.z * 255.0) as u8;
        self.data[idx + 3] = (colour.w * 255.0) as u8;
    }
    pub fn fill(&mut self, colour: Vec4) {
        for i in 0..self.w {
            for j in 0..self.h {
                self.set(i, j, colour);
            }
        }
    }
    pub fn set_square(&mut self, x: usize, y: usize, r: usize, colour: Vec4) {
        for j in (y as isize - r as isize).max(0) as usize..=(y + r).min(self.h - 1) {
            for i in (x as isize - r as isize).max(0) as usize..=(x + r).min(self.w - 1) {
                self.set(i, j, colour);
            }
        }
    }
    pub fn set_square_absolute(&mut self, p: Vec2, r: usize, colour: Vec4) {
        let x = (self.w as f32 * p.x) as usize;
        let y = (self.h as f32 * p.y) as usize;
        self.set_square(x, y, r, colour);
    }
    pub fn set_square_absolute_transform(&mut self, p: Vec2, tu: Vec3, tv: Vec3, tw: Vec3, r: usize, colour: Vec4) {
        let p = xform(p, tu, tv, tw);
        self.set_square_absolute(p, r, colour);
    }
    // draws a line between p1 and p2 of radius r pixels
    pub fn line_absolute(&mut self, p1: Vec2, p2: Vec2, colour: Vec4, r: usize) {
        let mut p = vec2(p1.x * self.w as f32, p1.y * self.h as f32);
        let p_dst = vec2(p2.x * self.w as f32, p2.y * self.h as f32);
        let u = p_dst - p;
        let d_final = u.magnitude();
        let mut d = 0.0;
        let dir = u.normalize();

        while d < d_final {
            self.set_square(p.x as usize, p.y as usize, r, colour);
            p += dir;
            d += dir.magnitude();
        }
    }
    pub fn circle(&mut self, x: usize, y: usize, r: usize, rr: usize, colour: Vec4) {
        let x = x as f32;
        let y = y as f32;
        let r = r as f32;
        
        let n = (PI * r) as usize;
        for i in 0..n {
            let theta = 2.0 * PI * i as f32 / n as f32;
            let px = x + r*theta.cos();
            let py = y + r*theta.sin();
            self.set_square(px as usize, py as usize, rr, colour);
        }
    }
    pub fn circle_absolute(&mut self, c: Vec2, r: f32, rr: usize, colour: Vec4) {
        let x = (self.w as f32 * c.x) as usize;
        let y = (self.h as f32 * c.y) as usize;
        let r = (r * self.w as f32) as usize;
        self.circle(x, y, r, rr, colour);
    }
    
    // provided pts are in some coordinate system where tu and tv will xform to 0..1
    pub fn circle_absolute_transform(&mut self, c: Vec2, tu: Vec3, tv: Vec3, tw: Vec3, r: f32, rr: usize, colour: Vec4) {
        let c = vec3(c.x, c.y, 1.0);
        let z = c.dot(tw);
        let c = vec2(c.dot(tu)/z, c.dot(tv)/z);
        let r = vec2(r, 0.0).dot(vec2(tu.x, tu.y));
        self.circle_absolute(c, r, rr, colour);
    }
    pub fn axis_transform(&mut self) {

    }
    pub fn plot(&mut self, y: &[f32], colour: Vec4, r: usize) {
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for i in 0..y.len() {
            if y[i] < min_y {
                min_y = y[i];
            }
            if y[i] > max_y {
                max_y = y[i];
            }
        }
        if max_y - min_y == 0.0 {
            max_y += 1.0;
            min_y -= -1.0;
        }
        // draw x axis
        let y0 = (0.0 - min_y) / (max_y - min_y);
        self.line_absolute(vec2(0.0, 1.0 - y0), vec2(1.0, 1.0 - y0), vec4(0.5, 0.5, 0.5, 1.0), 1);

        for i in 0..y.len() - 1 {
            let this_y = (y[i] - min_y) / (max_y - min_y);
            let next_y = (y[i+1] - min_y) / (max_y - min_y);
            let this_x = i as f32 / (y.len()-1) as f32;
            let next_x = (i+1) as f32 / (y.len()-1) as f32;
            self.line_absolute(vec2(this_x, 1.0 - this_y), vec2(next_x, 1.0 - next_y), colour, r);
        }
    }

    pub fn dump_to_file(&self, path: &str) {
        let path = Path::new(path);
        let file = File::create(path).unwrap();
        let ref mut buf_writer = BufWriter::new(file);

        let mut encoder = png::Encoder::new(buf_writer, self.w as u32, self.h as u32);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        // encoder.set_trns(vec!(0xFFu8, 0xFFu8, 0xFFu8)); // maybe dont need lol
        encoder.set_source_gamma(png::ScaledFloat::from_scaled(45455)); // 1.0 / 2.2, scaled by 100000
        encoder.set_source_gamma(png::ScaledFloat::new(1.0 / 2.2));     // 1.0 / 2.2, unscaled, but rounded
        let source_chromaticities = png::SourceChromaticities::new(     // Using unscaled instantiation here
            (0.31270, 0.32900),
            (0.64000, 0.33000),
            (0.30000, 0.60000),
            (0.15000, 0.06000)
        );
        encoder.set_source_chromaticities(source_chromaticities);
        let mut writer = encoder.write_header().unwrap();

        writer.write_image_data(&self.data).unwrap(); // Save
    }
}

#[test]
pub fn test_line_drawing() {
    let mut buf = ImageBuffer::new(1000, 1000);
    buf.line_absolute(vec2(0.1, 0.1), vec2(0.9, 0.9), vec4(1.0, 0.0, 0.0, 1.0), 3);
    buf.line_absolute(vec2(0.5, 0.5), vec2(0.1, 0.9), vec4(0.0, 0.0, 1.0, 1.0), 5);
    buf.dump_to_file("test.png");
}

#[test]
pub fn test_plot() {
    let mut buf = ImageBuffer::new(1000, 1000);
    buf.fill(vec4(1.0, 1.0, 1.0, 1.0));
    buf.plot(&vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 10.0, 3.0], vec4(1.0, 0.0, 0.0, 1.0), 2);
    buf.dump_to_file("test.png");
}

#[test]
pub fn test_circle() {
    let mut buf = ImageBuffer::new(1000, 1000);
    buf.fill(vec4(1.0, 1.0, 1.0, 1.0));
    buf.circle(500, 500, 400, 2, vec4(0.0, 0.0, 1.0, 1.0));
    buf.circle_absolute(vec2(0.5, 0.5), 0.5, 2, vec4(1.0, 0.0, 0.0, 1.0));
    // let tv = vec2(-1.1, -1.1);
    // let tu = vec2(1.1, 1.1);
    // let inv = matrix_inverse(tu, tv).unwrap();
    // let tu = inv.0;
    // let tv = inv.1;
    let tu = vec3(0.4, 0.0, 0.5);
    let tv = vec3(0.0, 0.4, 0.5);
    let tw = vec3(0.0, 0.0, 1.0);
    buf.circle_absolute_transform(vec2(0.0, 0.0), tu, tv, tw, 1.0, 2, vec4(0.0, 1.0, 0.0, 1.0));
    buf.dump_to_file("test.png");
}

#[test]
pub fn test_xform() {
    let tu = vec3(1.0, 0.0, 1.0);
    let tv = vec3(0.0, 1.0, 1.0);
    let tw = vec3(0.0, 0.0, 1.0);

    let p = vec2(0.0, 0.0);
    
    assert_eq!(xform(p, tu, tv, tw), vec2(1.0, 1.0));
    let p = vec2(1.0, 1.0);
    assert_eq!(xform(p, tu, tv, tw), vec2(2.0, 2.0));
}

pub fn xform(p: Vec2, tu: Vec3, tv: Vec3, tw: Vec3) -> Vec2 {
    let p = vec3(p.x, p.y, 1.0);
    let pt = vec2(p.dot(tu), p.dot(tv)) / p.dot(tw);
    vec2(pt.x, pt.y)
}