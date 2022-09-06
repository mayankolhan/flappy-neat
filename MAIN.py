import random

import pygame as pg
import neat
import os

## window config
WIN_W = 500
WIN_H = 936


## loading images
BIRD_IMGS = [pg.image.load(os.path.join("img" , "bird1.png")),pg.image.load(os.path.join("img" , "bird2.png")),pg.image.load(os.path.join("img" , "bird3.png"))]
PIPE_IMG = pg.image.load(os.path.join("img" , "pipe.png"))
GRND_IMG =pg.image.load(os.path.join("img" , "ground.png"))
BG_IMG = pg.image.load(os.path.join("img" , "bg.png"))

pg.font.init()
STAT_FONT = pg.font.SysFont("comicsans", 50)
##
pg.display.set_caption("JIGNESH BHAI")
class Bird:


    imgs= BIRD_IMGS
    max_rotation = 25
    ROT_VEL = 20
    ANIMATION_TIME =5

    def __init__(self,x,y):
        self.x = x
        self.y = y
        self. tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y

        self.img_count = 0
        self.img = self.imgs[0]

    def jump(self):
        self.vel = -10
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count+=0.02
        d = self.vel * self.tick_count + 1.5 * self.tick_count**2

        if d >= 16:
            d = 16


        self.y = self.y + d

        if d < 0 or self.y < self.height +50:
            if self.tilt < self.max_rotation:
                self.tilt = self.max_rotation
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count +=1

        if self.img_count < self.ANIMATION_TIME :
            self.img = self.imgs[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = self.imgs[1]

        elif self.img_count < self.ANIMATION_TIME * 3:
            self.img = self.imgs[2]
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.imgs[1]
        elif self.img_count < self.ANIMATION_TIME * 4 +1:
            self.img = self.imgs[0]
            self.img_count=0

        if self.tilt <= -80:
            self.img = self.imgs[1]
            self.img_count = self.ANIMATION_TIME*2

        rotated_img = pg.transform.rotate(self.img  , self.tilt)
        new_rect = rotated_img.get_rect(center = self.img.get_rect(topleft = (self.x,self.y)).center)

        win.blit(rotated_img , new_rect.topleft)

    def get_mask(self):
        return pg.mask.from_surface(self.img)

class Pipe:
    GAP =200
    VEL = 6

    def __init__(self,x ):
        self.x = x
        self.height =0
        self.gap =100

        self.top= 0
        self.bottom = 0
        self.PIPE_TOP  = pg.transform.flip(PIPE_IMG ,False ,True)
        self.PIPE_BOTTOM = PIPE_IMG
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randint(50,450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height+ self.GAP
    def move(self):
        self.x -= self.VEL

    def draw(self,win):
        win.blit(self.PIPE_TOP , (self.x, self.top))
        win.blit(self.PIPE_BOTTOM , (self.x , self.bottom))

    def collide(self ,bird):
        bird_mask = bird.get_mask()
        top_mask = pg.mask.from_surface(self.PIPE_TOP)
        bottom_mask= pg.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x , self.top - round(bird.y))
        bottom_offset = (self.x - bird.x , self.bottom -round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True
        return  False

class Base:
    VEl =5
    WIDHT = GRND_IMG.get_width()
    IMG = GRND_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDHT

    def move(self):
        self.x1 -= self.VEl
        self.x2 -= self.VEl

        if self.x1 + self.WIDHT < 0:
            self.x1 = self.x2 +self.WIDHT
        if self.x2 + self.WIDHT < 0:
            self.x2  = self.x1 +self.WIDHT

    def draw(self,win):
        win.blit(self.IMG ,(self.x1 , self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win,birds,pipes ,base ,score):
    win.blit(BG_IMG,(0,0))
    for pipe in pipes:
        pipe.draw(win)
    text = STAT_FONT.render("SCORE : " +str(score),1,(255,255,255))
    win.blit(text, (WIN_W -10  - text.get_width() ,10))
    base.draw(win)
    for bird in birds:
        bird.draw(win)
    pg.display.update()

def main(genomes , config):
    nets = []
    ge = []
    jignesh_family = []
    for _,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        jignesh_family.append(Bird(50,350))
        g.fitness= 0
        ge.append(g)

    disp = pg.display.set_mode((WIN_W, WIN_H))

    pipes = [Pipe(350)]
    base= Base(730)
    run =True
    fps = pg.time.Clock()
    score= 0

    while run :
        fps.tick(30)
        for event in pg.event.get():
            if event.type ==  pg.QUIT:
                run =False
                pg.quit()
       # jignesh.move()
        pipe_ind=0
        if len(jignesh_family) > 0:
            if len(pipes) >1 and jignesh_family[0].x > pipes[0].x +pipes[0].PIPE_TOP.get_width():
                pipe_ind =1
        else:
            run =False
            break
        for x , jignesh_relative in enumerate(jignesh_family):

            ge[x].fitness+=0.1
            output = nets[x].activate((jignesh_relative.y , abs(jignesh_relative.y - pipes[pipe_ind].height) , abs(jignesh_relative.y - pipes[pipe_ind].bottom)))

            if output[0] == 1:
                jignesh_relative.jump()
            jignesh_relative.move()
        add_pipe =False
        rem= []
        for pipe in pipes:
            for x,jignesh_relative in enumerate(jignesh_family):
                if pipe.collide(jignesh_relative):
                    ge[x].fitness -=1
                    jignesh_family.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < jignesh_relative.x:
                    pipe.passed = True
                    add_pipe = True
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)
            pipe.move()
        if add_pipe:
            score+=1
            for g in ge:
                g.fitness+=5
            pipes.append(Pipe(700))
        for r in rem:
            pipes.remove(r)
        for x, jignesh_relative in enumerate(jignesh_family):
            if jignesh_relative.y + jignesh_relative.img.get_height() >= 738 or jignesh_relative.y < 0:
                jignesh_family.pop(x)
                nets.pop(x)
                ge.pop(x)


        base.move()
        draw_window(disp,jignesh_family,pipes,base,score)



def run(config_path):
    config = neat.config.Config(neat.DefaultGenome ,neat.DefaultReproduction ,
                                neat.DefaultSpeciesSet , neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main,50)



local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir , "config_feed.txt")
run(config_path)



