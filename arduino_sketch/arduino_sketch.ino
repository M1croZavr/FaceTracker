#include <Servo.h>


int pos = 0;
int current_pos = 0;
int servoPin = 9;
int servoDelay = 15;
int LED1 = 13;
int LED2 = 12;
int LED3 = 11;

Servo myServo;

void setup() {
  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  pinMode(LED3, OUTPUT);
  Serial.begin(9600);
  myServo.attach(servoPin);
  myServo.write(0);
}

void loop() {
  // if (current_pos = 0){
  //   myServo.write(current_pos);
  // }
  while (Serial.available() > 0){
    digitalWrite(LED3, HIGH);
    pos = Serial.parseInt();
    // if (pos = pos){
    if (90 < pos){
      digitalWrite(LED1, LOW);
      digitalWrite(LED2, HIGH);
      myServo.write(pos);
      // for (int i = current_pos; i < pos; i += 1){
      //   current_pos += 1;
      //   myServo.write(current_pos);
      //   delay(1);
      // }
    }
    else if (90 >= pos){
      digitalWrite(LED2, LOW);
      digitalWrite(LED1, HIGH);
      myServo.write(pos);
      // for (int i = current_pos; i > pos; i -= 1){
      //   current_pos -= 1;
      //   myServo.write(current_pos);
      //   delay(1);
      // }
    }
    // Serial.println(current_pos);
  
    // myServo.write(pos);
    // Serial.println(pos);
    // }
    // delay(servoDelay);
  }
}


// #include <Servo.h>

// Servo myservo;  // create servo object to control a servo

// int x = 0;
// int pos = 0;
// void setup() {
//   // initialize digital pin LED_BUILTIN as an output.
//   // pinMode(LED_BUILTIN, OUTPUT);
//   Serial.begin(9600);
//   Serial.setTimeout(1);
//   myservo.attach(9);
// }

// // the loop function runs over and over again forever
// void loop() {
//   // digitalWrite(LED_BUILTIN, LOW);
//   myservo.write(0);
//   if (!Serial.available()){
//   x = Serial.readString().toInt();
//   for (pos = 0; pos <= x; pos += 1) { // goes from 0 degrees to 180 degrees
//     // in steps of 1 degree
//     myservo.write(pos);              // tell servo to go to position in variable 'pos'
//     delay(15);                       // waits 15 ms for the servo to reach the position
//   }
//   }
//   // myservo.write(x);
//   // Serial.print(x);
//   // delay(5000);
//   // x = Serial.readString().toInt();
//   // digitalWrite(LED_BUILTIN, HIGH);
//   // delay(5000);
// }

// int x;

// void setup() {
//   Serial.begin(115200);
//   Serial.setTimeout(1);
// }

// void loop() {
//   while (!Serial.available());
//   x = Serial.readString().toInt();
//   Serial.print(x + 1);
// }
// #include <Servo.h>

// Servo myservo;

// void setup() {
//   // put your setup code here, to run once:
//   myservo.attach(9);
// }

// void loop() {
//   // put your main code here, to run repeatedly:
//   myservo.write(0);
//   delay(1000);
//   myservo.write(180);
//   delay(1000);
// }
