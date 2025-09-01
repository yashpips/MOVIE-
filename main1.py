from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, ForeignKey, Text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import json
import threading
import uuid

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./movie_booking.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Thread-safe booking lock
booking_lock = threading.Lock()

# Database Models
class Movie(Base):
    __tablename__ = "movies"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    duration = Column(Integer)
    genre = Column(String)
    price = Column(Float)
    
    shows = relationship("Show", back_populates="movie")

class Theater(Base):
    __tablename__ = "theaters"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    location = Column(String)
    
    halls = relationship("Hall", back_populates="theater")

class Hall(Base):
    __tablename__ = "halls"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    theater_id = Column(Integer, ForeignKey("theaters.id"))
    row_config = Column(Text)  # JSON: {"rows": [7, 8, 6, 9]} - seats per row
    
    theater = relationship("Theater", back_populates="halls")
    shows = relationship("Show", back_populates="hall")

class Show(Base):
    __tablename__ = "shows"
    
    id = Column(Integer, primary_key=True, index=True)
    movie_id = Column(Integer, ForeignKey("movies.id"))
    hall_id = Column(Integer, ForeignKey("halls.id"))
    show_time = Column(DateTime)
    
    movie = relationship("Movie", back_populates="shows")
    hall = relationship("Hall", back_populates="shows")
    bookings = relationship("Booking", back_populates="show")

class Booking(Base):
    __tablename__ = "bookings"
    
    id = Column(Integer, primary_key=True, index=True)
    show_id = Column(Integer, ForeignKey("shows.id"))
    booking_reference = Column(String, unique=True, index=True)
    seats = Column(Text)  # JSON: [{"row": 1, "seat": 2}, ...]
    total_amount = Column(Float)
    booking_time = Column(DateTime, default=datetime.utcnow)
    group_size = Column(Integer)
    
    show = relationship("Show", back_populates="bookings")

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class MovieCreate(BaseModel):
    title: str
    description: str = ""
    duration: int = 120
    genre: str = ""
    price: float

class MovieUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    duration: Optional[int] = None
    genre: Optional[str] = None
    price: Optional[float] = None

class TheaterCreate(BaseModel):
    name: str
    location: str

class TheaterUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None

class HallCreate(BaseModel):
    name: str
    theater_id: int
    rows: List[int]  # [7, 8, 6, 9] - seats per row (min 6, exactly 6 isle seats)
    
    @validator('rows')
    def validate_rows(cls, v):
        for i, seats in enumerate(v):
            if seats < 6:
                raise ValueError(f"Row {i+1} must have at least 6 seats")
            # Each row must have exactly 6 isle seats (3 columns)
            if (seats - 6) % 3 != 0:
                raise ValueError(f"Row {i+1} must have 6 base seats + multiples of 3 additional seats")
        return v

class HallUpdate(BaseModel):
    name: Optional[str] = None
    rows: Optional[List[int]] = None

class ShowCreate(BaseModel):
    movie_id: int
    hall_id: int
    show_time: datetime

class BookingRequest(BaseModel):
    movie_id: int
    show_time: datetime
    theater_id: int
    group_size: int

# FastAPI app
app = FastAPI(title="Movie Ticket Booking API")

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper functions
def get_booked_seats(db: Session, show_id: int) -> set:
    """Get all booked seats for a show"""
    bookings = db.query(Booking).filter(Booking.show_id == show_id).all()
    booked_seats = set()
    
    for booking in bookings:
        seats = json.loads(booking.seats)
        for seat in seats:
            booked_seats.add((seat['row'], seat['seat']))
    
    return booked_seats

def find_consecutive_seats(rows_config: List[int], booked_seats: set, group_size: int) -> Optional[List[Dict]]:
    """Find consecutive seats for a group - respects 6 isle seats rule"""
    for row_idx, total_seats in enumerate(rows_config):
        row_num = row_idx + 1
        
        # For each row, check all possible starting positions
        for start_seat in range(1, total_seats - group_size + 2):
            consecutive_seats = []
            
            # Check if we can book group_size consecutive seats
            valid_booking = True
            for offset in range(group_size):
                seat_num = start_seat + offset
                if (row_num, seat_num) in booked_seats:
                    valid_booking = False
                    break
                consecutive_seats.append({"row": row_num, "seat": seat_num})
            
            if valid_booking:
                return consecutive_seats
    
    return None

# MOVIE CRUD APIs
@app.post("/movies")
def create_movie(movie: MovieCreate, db: Session = Depends(get_db)):
    db_movie = Movie(**movie.dict())
    db.add(db_movie)
    db.commit()
    db.refresh(db_movie)
    return db_movie

@app.get("/movies")
def get_movies(db: Session = Depends(get_db)):
    return db.query(Movie).all()

@app.get("/movies/{movie_id}")
def get_movie(movie_id: int, db: Session = Depends(get_db)):
    movie = db.query(Movie).filter(Movie.id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie

@app.put("/movies/{movie_id}")
def update_movie(movie_id: int, movie: MovieUpdate, db: Session = Depends(get_db)):
    db_movie = db.query(Movie).filter(Movie.id == movie_id).first()
    if not db_movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    for key, value in movie.dict(exclude_unset=True).items():
        setattr(db_movie, key, value)
    
    db.commit()
    db.refresh(db_movie)
    return db_movie

@app.delete("/movies/{movie_id}")
def delete_movie(movie_id: int, db: Session = Depends(get_db)):
    db_movie = db.query(Movie).filter(Movie.id == movie_id).first()
    if not db_movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    db.delete(db_movie)
    db.commit()
    return {"message": "Movie deleted"}

# THEATER CRUD APIs
@app.post("/theaters")
def create_theater(theater: TheaterCreate, db: Session = Depends(get_db)):
    db_theater = Theater(**theater.dict())
    db.add(db_theater)
    db.commit()
    db.refresh(db_theater)
    return db_theater

@app.get("/theaters")
def get_theaters(db: Session = Depends(get_db)):
    return db.query(Theater).all()

@app.get("/theaters/{theater_id}")
def get_theater(theater_id: int, db: Session = Depends(get_db)):
    theater = db.query(Theater).filter(Theater.id == theater_id).first()
    if not theater:
        raise HTTPException(status_code=404, detail="Theater not found")
    return theater

@app.put("/theaters/{theater_id}")
def update_theater(theater_id: int, theater: TheaterUpdate, db: Session = Depends(get_db)):
    db_theater = db.query(Theater).filter(Theater.id == theater_id).first()
    if not db_theater:
        raise HTTPException(status_code=404, detail="Theater not found")
    
    for key, value in theater.dict(exclude_unset=True).items():
        setattr(db_theater, key, value)
    
    db.commit()
    db.refresh(db_theater)
    return db_theater

@app.delete("/theaters/{theater_id}")
def delete_theater(theater_id: int, db: Session = Depends(get_db)):
    db_theater = db.query(Theater).filter(Theater.id == theater_id).first()
    if not db_theater:
        raise HTTPException(status_code=404, detail="Theater not found")
    
    db.delete(db_theater)
    db.commit()
    return {"message": "Theater deleted"}

# HALL CRUD APIs
@app.post("/halls")
def create_hall(hall: HallCreate, db: Session = Depends(get_db)):
    # Check theater exists
    theater = db.query(Theater).filter(Theater.id == hall.theater_id).first()
    if not theater:
        raise HTTPException(status_code=404, detail="Theater not found")
    
    db_hall = Hall(
        name=hall.name,
        theater_id=hall.theater_id,
        row_config=json.dumps(hall.rows)
    )
    db.add(db_hall)
    db.commit()
    db.refresh(db_hall)
    return {
        "id": db_hall.id,
        "name": db_hall.name,
        "theater_id": db_hall.theater_id,
        "rows": json.loads(db_hall.row_config)
    }

@app.get("/halls")
def get_halls(theater_id: Optional[int] = None, db: Session = Depends(get_db)):
    query = db.query(Hall)
    if theater_id:
        query = query.filter(Hall.theater_id == theater_id)
    
    halls = query.all()
    result = []
    for hall in halls:
        result.append({
            "id": hall.id,
            "name": hall.name,
            "theater_id": hall.theater_id,
            "rows": json.loads(hall.row_config)
        })
    return result

@app.get("/halls/{hall_id}")
def get_hall(hall_id: int, db: Session = Depends(get_db)):
    hall = db.query(Hall).filter(Hall.id == hall_id).first()
    if not hall:
        raise HTTPException(status_code=404, detail="Hall not found")
    return {
        "id": hall.id,
        "name": hall.name,
        "theater_id": hall.theater_id,
        "rows": json.loads(hall.row_config)
    }

@app.put("/halls/{hall_id}")
def update_hall(hall_id: int, hall: HallUpdate, db: Session = Depends(get_db)):
    db_hall = db.query(Hall).filter(Hall.id == hall_id).first()
    if not db_hall:
        raise HTTPException(status_code=404, detail="Hall not found")
    
    if hall.name:
        db_hall.name = hall.name
    if hall.rows:
        # Validate rows
        for i, seats in enumerate(hall.rows):
            if seats < 6:
                raise HTTPException(status_code=400, detail=f"Row {i+1} must have at least 6 seats")
            if (seats - 6) % 3 != 0:
                raise HTTPException(status_code=400, detail=f"Row {i+1} must have 6 base + multiples of 3 seats")
        db_hall.row_config = json.dumps(hall.rows)
    
    db.commit()
    db.refresh(db_hall)
    return {
        "id": db_hall.id,
        "name": db_hall.name,
        "theater_id": db_hall.theater_id,
        "rows": json.loads(db_hall.row_config)
    }

@app.delete("/halls/{hall_id}")
def delete_hall(hall_id: int, db: Session = Depends(get_db)):
    db_hall = db.query(Hall).filter(Hall.id == hall_id).first()
    if not db_hall:
        raise HTTPException(status_code=404, detail="Hall not found")
    
    db.delete(db_hall)
    db.commit()
    return {"message": "Hall deleted"}

# HALL LAYOUT WITH SEAT STATUS
@app.get("/halls/{hall_id}/layout")
def get_hall_layout(hall_id: int, show_id: Optional[int] = None, db: Session = Depends(get_db)):
    hall = db.query(Hall).filter(Hall.id == hall_id).first()
    if not hall:
        raise HTTPException(status_code=404, detail="Hall not found")
    
    rows = json.loads(hall.row_config)
    booked_seats = set()
    
    if show_id:
        booked_seats = get_booked_seats(db, show_id)
    
    # Create seat layout with exactly 6 isle seats (3 columns) per row
    layout = []
    for row_idx, total_seats in enumerate(rows):
        row_num = row_idx + 1
        row_layout = {
            "row": row_num,
            "total_seats": total_seats,
            "columns": [
                {"seats": [], "is_isle": False},  # Column 1
                {"seats": [], "is_isle": False},  # Column 2  
                {"seats": [], "is_isle": False}   # Column 3
            ]
        }
        
        # Distribute seats across 3 columns (6 isle seats minimum)
        seats_per_column = total_seats // 3
        extra_seats = total_seats % 3
        
        seat_num = 1
        for col_idx in range(3):
            col_seats = seats_per_column + (1 if col_idx < extra_seats else 0)
            
            for _ in range(col_seats):
                is_booked = (row_num, seat_num) in booked_seats
                row_layout["columns"][col_idx]["seats"].append({
                    "seat": seat_num,
                    "booked": is_booked
                })
                seat_num += 1
            
            row_layout["columns"][col_idx]["is_isle"] = True  # All columns have isle access
        
        layout.append(row_layout)
    
    return {
        "hall_id": hall_id,
        "layout": layout,
        "empty_seats": sum(len(rows) for rows in [r["columns"] for r in layout]) - len(booked_seats),
        "booked_seats": len(booked_seats)
    }

# SHOW CRUD APIs
@app.post("/shows")
def create_show(show: ShowCreate, db: Session = Depends(get_db)):
    # Validate movie and hall exist
    movie = db.query(Movie).filter(Movie.id == show.movie_id).first()
    hall = db.query(Hall).filter(Hall.id == show.hall_id).first()
    
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    if not hall:
        raise HTTPException(status_code=404, detail="Hall not found")
    
    db_show = Show(**show.dict())
    db.add(db_show)
    db.commit()
    db.refresh(db_show)
    return db_show

@app.get("/shows")
def get_shows(movie_id: Optional[int] = None, theater_id: Optional[int] = None, db: Session = Depends(get_db)):
    query = db.query(Show)
    
    if movie_id:
        query = query.filter(Show.movie_id == movie_id)
    if theater_id:
        query = query.join(Hall).filter(Hall.theater_id == theater_id)
    
    return query.all()

@app.get("/shows/{show_id}")
def get_show(show_id: int, db: Session = Depends(get_db)):
    show = db.query(Show).filter(Show.id == show_id).first()
    if not show:
        raise HTTPException(status_code=404, detail="Show not found")
    return show

# GROUP BOOKING API (Main Feature)
@app.post("/bookings")
def book_tickets(booking: BookingRequest, db: Session = Depends(get_db)):
    with booking_lock:  # Thread-safe booking
        # Find the specific show
        show = db.query(Show).join(Hall).filter(
            Show.movie_id == booking.movie_id,
            Show.show_time == booking.show_time,
            Hall.theater_id == booking.theater_id
        ).first()
        
        if not show:
            raise HTTPException(status_code=404, detail="Show not found for given movie, time, and theater")
        
        # Get hall configuration
        hall_rows = json.loads(show.hall.row_config)
        booked_seats = get_booked_seats(db, show.id)
        
        # Try to find consecutive seats
        selected_seats = find_consecutive_seats(hall_rows, booked_seats, booking.group_size)
        
        if not selected_seats:
            # Find alternative shows
            alternatives = find_alternative_shows(db, booking)
            raise HTTPException(status_code=400, detail={
                "message": f"Cannot book {booking.group_size} seats together",
                "alternatives": alternatives
            })
        
        # Create booking
        booking_ref = str(uuid.uuid4())[:8].upper()
        total_amount = booking.group_size * show.movie.price
        
        db_booking = Booking(
            show_id=show.id,
            booking_reference=booking_ref,
            seats=json.dumps(selected_seats),
            total_amount=total_amount,
            group_size=booking.group_size
        )
        
        db.add(db_booking)
        db.commit()
        
        return {
            "booking_reference": booking_ref,
            "show_id": show.id,
            "movie": show.movie.title,
            "theater": show.hall.theater.name,
            "hall": show.hall.name,
            "show_time": show.show_time,
            "seats": selected_seats,
            "total_amount": total_amount,
            "group_size": booking.group_size
        }

def find_alternative_shows(db: Session, booking: BookingRequest) -> List[Dict]:
    """Find alternative shows with available consecutive seats"""
    # Get other shows for same movie on same day
    target_date = booking.show_time.date()
    
    alternative_shows = db.query(Show).join(Hall).filter(
        Show.movie_id == booking.movie_id,
        func.date(Show.show_time) == target_date,
        Show.show_time != booking.show_time,
        Hall.theater_id == booking.theater_id
    ).all()
    
    # Also check other theaters for same movie
    other_theater_shows = db.query(Show).join(Hall).filter(
        Show.movie_id == booking.movie_id,
        func.date(Show.show_time) == target_date,
        Hall.theater_id != booking.theater_id
    ).all()
    
    all_alternatives = alternative_shows + other_theater_shows
    
    alternatives = []
    for show in all_alternatives:
        hall_rows = json.loads(show.hall.row_config)
        booked_seats = get_booked_seats(db, show.id)
        
        if find_consecutive_seats(hall_rows, booked_seats, booking.group_size):
            alternatives.append({
                "show_id": show.id,
                "movie": show.movie.title,
                "theater": show.hall.theater.name,
                "hall": show.hall.name,
                "time": show.show_time.isoformat(),
                "available": True
            })
    
    return alternatives[:5]  # Return top 5 alternatives

@app.get("/bookings/{booking_reference}")
def get_booking(booking_reference: str, db: Session = Depends(get_db)):
    booking = db.query(Booking).filter(Booking.booking_reference == booking_reference).first()
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    return {
        "booking_reference": booking.booking_reference,
        "show_id": booking.show_id,
        "movie": booking.show.movie.title,
        "theater": booking.show.hall.theater.name,
        "show_time": booking.show.show_time,
        "seats": json.loads(booking.seats),
        "total_amount": booking.total_amount,
        "booking_time": booking.booking_time
    }

# ANALYTICS APIs
@app.get("/analytics/movie/{movie_id}")
def movie_analytics(
    movie_id: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Booking).join(Show).filter(Show.movie_id == movie_id)
    
    if start_date:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        query = query.filter(Booking.booking_time >= start)
    
    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        query = query.filter(Booking.booking_time < end)
    
    bookings = query.all()
    
    total_tickets = sum(b.group_size for b in bookings)
    total_gmv = sum(b.total_amount for b in bookings)
    
    # Get movie title
    movie = db.query(Movie).filter(Movie.id == movie_id).first()
    movie_title = movie.title if movie else "Unknown"
    
    return {
        "movie_id": movie_id,
        "movie_title": movie_title,
        "period": f"{start_date or 'all time'} to {end_date or 'now'}",
        "total_tickets_booked": total_tickets,
        "total_gmv": total_gmv,
        "total_bookings": len(bookings),
        "average_group_size": round(total_tickets / len(bookings), 1) if bookings else 0
    }

@app.get("/analytics/theater/{theater_id}")  
def theater_analytics(theater_id: int, db: Session = Depends(get_db)):
    bookings = db.query(Booking).join(Show).join(Hall).filter(Hall.theater_id == theater_id).all()
    
    movie_breakdown = {}
    total_revenue = 0
    total_tickets = 0
    
    for booking in bookings:
        movie_title = booking.show.movie.title
        if movie_title not in movie_breakdown:
            movie_breakdown[movie_title] = {"tickets": 0, "revenue": 0}
        
        movie_breakdown[movie_title]["tickets"] += booking.group_size
        movie_breakdown[movie_title]["revenue"] += booking.total_amount
        total_revenue += booking.total_amount
        total_tickets += booking.group_size
    
    return {
        "theater_id": theater_id,
        "total_tickets": total_tickets,
        "total_revenue": total_revenue,
        "movies": movie_breakdown
    }

@app.get("/")
def home():
    return {
        "message": "Movie Booking API - All Features Ready!",
        "features": [
            "✅ CRUD for Movies, Theaters, Halls", 
            "✅ Group booking with consecutive seats",
            "✅ 6 isle seats (3 columns) per row enforced",
            "✅ Alternative suggestions when full",
            "✅ Thread-safe booking (no double booking)",
            "✅ Analytics with GMV tracking"
        ],
        "test_at": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)