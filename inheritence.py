class Parent():
    def __init__(self,last_name,eye_color):
        self.last_name = last_name
        self.eye_color = eye_color

    def show_eye_color(self):
        print(self.eye_color)

class Child(Parent): #using Parent in parenthesis means its inheriting from the Parent class
    def __init__(self,last_name, eye_color,num_of_toys):
        Parent.__init__(self,last_name,eye_color)
        self.num_of_toys = num_of_toys

    #method overriding
    def show_eye_color(self):
        print(self.last_name + " has " + self.eye_color + " eyes.")

if __name__ == '__main__':
    N = Parent("S","Dark Brown")
    N.show_eye_color()

    A = Child("S","Brown",5)
    A.show_eye_color()

