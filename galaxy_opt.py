import taichi as ti
import handy_shader_functions as hsf

ti.init(arch=ti.gpu)
res_x = 1200
res_y = 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

@ti.kernel
def paint(iTime: ti.f32):
    for i, j in pixels:
        col = ti.Vector([0.0, 0.0, 0.0])
        uv = ti.Vector([float(i) / res_x, float(j) / res_y]) - 0.5
        lens = uv.norm()
        p = ti.Vector([0.0, 0.0, 0.0])
        t = 0.01 * iTime
        time = t + (5.0 + ti.sin(t)) * 0.11 / (lens + 0.07)
        si = ti.sin(time)
        co = ti.cos(time)
        ma = ti.Matrix([[co, si], [-si, co]])
        uv = ma @ uv
        v1 = v2 = 0.0
        for k1 in range(100):
            p = 0.035 * float(k1) * ti.Vector([uv[0], uv[1], 1.0])
            p += ti.Vector([0.22, 0.3, -1.5 - ti.sin(t * 1.3) * 0.1])
            for k2 in range(8):
                len2 = p.dot(p)
                p = ti.abs(p) / len2 - 0.659
            p2 = (p.dot(p)) * 0.0015
            v1 += p2 * (1.8 + ti.sin(lens * 13.0 + 0.5 - t * 2.0))
            v2 += p2 * (1.5 + ti.sin(lens * 13.5 + 2.2 - t * 3.0))
        c = p.norm() * 0.175
        v1 *= hsf.smoothstep(0.7, 0.0, lens)
        v2 *= hsf.smoothstep(0.6, 0.0, lens)
        v3 = hsf.smoothstep(0.15, 0.0, lens)
        col[0] = c
        col[1] = (v1 + c) * 0.25
        col[2] = v2
        col = col + v3 * 0.9
        pixels[i, j] = col

gui = ti.GUI("quiz", (res_x, res_y))
t = 0.0
while True:
    t += 0.05
    paint(t)
    gui.set_image(pixels)
    gui.show()