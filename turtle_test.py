import turtle

def draw_flower():
    for angle in range(36):
        for i in range(2):
            draw_turtle.forward(100)
            draw_turtle.right(45)
            draw_turtle.forward(100)
            draw_turtle.right(135)
        draw_turtle.right(10)

    draw_turtle.right(90)
    draw_turtle.forward(300)

if __name__ == '__main__':
    # open a window for drawing
    draw_window = turtle.Screen()
    draw_window.bgcolor('yellow')

    # draw square on window
    draw_turtle = turtle.Turtle()
    draw_turtle.shape('turtle')
    draw_turtle.color('blue','orange')
    draw_turtle.speed(2)

    draw_turtle.begin_fill()
    draw_flower()
    draw_turtle.end_fill()
    
    draw_window.exitonclick()
